"""
Resources:
- [CMU-11-667 Large Language Models: Methods and Applications](https://cmu-llms.org/)
- [CMU-11-711 Advanced NLP](https://phontron.com/class/anlp2024/)
- [ACL 2023 Tutorial: Retrieval-based Language Models and Applications](https://acl2023-retrieval-lm.github.io/)
- [Build a RAG agent with LangChain](https://docs.langchain.com/oss/python/langchain/rag)
- [LlamaIndex RAG](https://developers.llamaindex.ai/python/framework/use_cases/q_and_a/)
- [llama-cookbook](https://github.com/meta-llama/llama-cookbook/tree/main)
- [cs-self-learning](https://csdiy.wiki/en/)
- [DPR](https://github.com/facebookresearch/DPR/tree/main)
- [Karpukhin et al., 2020](https://arxiv.org/pdf/2004.04906)
"""

from pathlib import Path
import sys

project_dir = Path(__file__).parent.parent
project_path = project_dir.as_posix()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import os
import time
from typing import Dict, Union
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm, trange
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import wandb

try:
    from datasets import DatasetDict
except:
    pass


# ----- dataset utils -----
def save_jsonl(lines: Union[list[dict], pd.DataFrame], path, overwrite=False):
    if isinstance(lines, pd.DataFrame):
        lines = lines.to_dict(orient="records")

    path = Path(path)
    if not overwrite:
        assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, line in enumerate(lines):
            json_line = json.dumps(line, ensure_ascii=False, separators=(",", ":"))
            if i == 0:
                f.write(json_line)
            else:
                f.write("\n" + json_line)
    print(f"Saved {len(lines)} lines to {path.as_posix()}")


def load_jsonl(path):
    data = pd.read_json(path_or_buf=path, lines=True)
    return data


def make_huatuo_512(dataset: "DatasetDict", key="test", overwrite=False):
    np.random.seed(42)
    assert key in ["train", "test", "validation"]
    path = project_dir / "dataset" / "huatuo_encyclopedia_qa_512" / f"{key}.jsonl"
    questions = list(dataset[key]["questions"])
    for i in range(len(questions)):
        assert len(questions[i]) == 1
        # choose one question randomly because they are similar
        questions[i] = np.random.choice(questions[i][0]).item()
    answers = list(dataset[key]["answers"])
    for i in range(len(answers)):
        assert len(answers[i]) == 1
        answers[i] = answers[i][0]
    indexes = []
    for i in range(len(questions)):
        if len(questions[i]) < 512 and len(answers[i]) < 512:
            indexes.append(i)
    df = pd.DataFrame(
        {
            "question": [questions[i] for i in indexes],
            "answer": [answers[i] for i in indexes],
        }
    )
    save_jsonl(df, path, overwrite=overwrite)


def load_dataset(tokenizer: PreTrainedTokenizerBase):
    path = project_dir / "dataset" / "huatuo_encyclopedia_qa_512"
    train_dataset = DPRDataset(path / "train.jsonl", tokenizer)
    test_dataset = DPRDataset(path / "test.jsonl", tokenizer)
    return train_dataset, test_dataset


class DPRDataset(Dataset):
    def __init__(
        self, file_path: str, tokenizer: PreTrainedTokenizerBase, max_length: int = 512
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_jsonl(file_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples.iloc[index]
        encoding = self.tokenizer(
            self.tokenizer.bos_token + sample["question"],
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze()
        attention_mask = encoding.attention_mask.squeeze()

        X = torch.tensor(input_ids, dtype=torch.long)
        attn_mask = torch.tensor(attention_mask, dtype=torch.long)
        return X, attn_mask


# ----- model -----
class Retriever(nn.Module):
    """
    Adapted from CMU-11-667 (https://cmu-llms.org/) EncoderModel
    """

    def __init__(self, encoder: PreTrainedModel, temperature: float = 1.0):
        super().__init__()
        self.config = encoder.config
        self.encoder = encoder
        self.temperature = temperature

    def forward(
        self,
        query_inputs: Dict[str, torch.Tensor] = None,
        passage_inputs: Dict[str, torch.Tensor] = None,
    ):
        q_reps = (
            self.encode(query_inputs) if query_inputs else None
        )  # (n_queries, hidden_dim)
        p_reps = (
            self.encode(passage_inputs) if passage_inputs else None
        )  # (n_passages, hidden_dim)

        # for inference
        if q_reps is None or p_reps is None:
            return q_reps, p_reps

        similarity = (
            torch.matmul(q_reps, p_reps.T) / self.temperature
        )  # (n_queries, n_passages)
        target = torch.arange(len(q_reps)).to(q_reps.device)  # (n_queries,)
        loss = F.cross_entropy(similarity, target)
        return loss, similarity

    def encode(self, inputs: Dict[str, torch.Tensor]):
        hidden_states = self.encoder(**inputs, output_hidden_states=True).hidden_states
        hidden_states = hidden_states[-1]  # (batch_size, seq_len, hidden_dim)
        return self.pooling(hidden_states, inputs["attention_mask"])

    def pooling(self, last_hidden_state, attention_mask):
        """
        last_hidden_state: (batch_size, seq_len, hidden_dim)
        attention_mask: (batch_size, seq_len)
        reps: (batch_size, hidden_dim)
        """
        attention_mask = torch.where(
            attention_mask == 1,
            attention_mask + torch.arange(attention_mask.shape[1]),
            0,
        )  # (batch_size, seq_len)
        last_nonzero_indices = attention_mask.argmax(
            dim=1, keepdim=True
        )  # (batch_size, 1)
        last_token_indices = last_nonzero_indices.unsqueeze(-1).expand(
            -1, -1, last_hidden_state.shape[-1]
        )  # (batch_size, 1, hidden_dim)
        last_hidden_state = torch.gather(
            last_hidden_state, 1, last_token_indices
        )  # (batch_size, 1, hidden_dim)
        last_hidden_state = last_hidden_state.squeeze(1)  # (batch_size, hidden_dim)
        reps = F.normalize(last_hidden_state, p=2, dim=1)
        return reps
