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

from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from multiprocessing import Pool
import os
import re
import time
from typing import List, Union
import warnings

import datasets
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm, trange
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase
import wandb

from model.model_minimind import MiniMindConfig
from trainer.trainer_utils import (
    Logger,
    MiniMindForCausalLM,
    SkipBatchSampler,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings("ignore")


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


def load_nq_dataset(max_query_len=64, max_passage_len=512):
    dataset = datasets.load_dataset("sentence-transformers/natural-questions")
    dataset = dataset.filter(
        lambda x: len(x["query"]) <= max_query_len
        and len(x["answer"]) <= max_passage_len
    )
    dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
    return dataset


class DPRDataset(Dataset):
    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_query_len=64,
        max_passage_len: int = 512,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        def encode(key, max_length):
            inputs = self.tokenizer(
                sample[key],
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                padding_side="right",
                padding="max_length",
                return_token_type_ids=False,
            )
            inputs["input_ids"] = inputs["input_ids"].squeeze(0)
            inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
            return inputs

        query_inputs = encode("query", self.max_query_len)
        passage_inputs = encode("answer", self.max_passage_len)
        return query_inputs, passage_inputs


# ----- model -----
class Encoder(nn.Module):
    """
    Adapted from CMU-11-667 (https://cmu-llms.org/) EncoderModel
    """

    def __init__(self, encoder: PreTrainedModel, temperature: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

    def forward(
        self,
        query_inputs: BatchEncoding = None,
        passage_inputs: BatchEncoding = None,
    ):
        q_embeddings = (
            self.encode(query_inputs) if query_inputs else None
        )  # (n_queries, hidden_dim)
        p_embeddings = (
            self.encode(passage_inputs) if passage_inputs else None
        )  # (n_passages, hidden_dim)

        # for inference
        if q_embeddings is None or p_embeddings is None:
            return q_embeddings, p_embeddings

        similarity = (
            torch.matmul(q_embeddings, p_embeddings.T) / self.temperature
        )  # (n_queries, n_passages)
        target = torch.arange(len(q_embeddings)).to(q_embeddings.device)  # (n_queries,)
        loss = F.cross_entropy(similarity, target)
        return loss, similarity

    def encode(self, inputs: BatchEncoding):
        hidden_states = self.encoder(
            **inputs
        ).last_hidden_state  # (batch_size, seq_len, hidden_dim)
        return self.pooling(hidden_states, inputs["attention_mask"])

    def pooling(self, last_hidden_state, attention_mask):
        """
        last_hidden_state: (batch_size, seq_len, hidden_dim)
        attention_mask: (batch_size, seq_len)
        reps: (batch_size, hidden_dim)
        """
        attention_mask = torch.where(
            attention_mask == 1,
            attention_mask
            + torch.arange(attention_mask.shape[1]).to(attention_mask.device),
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


@torch.inference_mode()
def similarity_bert(model_bert, queries: List[str], passages: List[str]):
    if isinstance(queries, str):
        queries = [queries]
    if isinstance(passages, str):
        passages = [passages]
    q_embeddings = model_bert.encode(queries)  # (n_queries, hidden_dim)
    p_embeddings = model_bert.encode(passages)  # (n_passages, hidden_dim)
    similarity = model_bert.similarity(
        q_embeddings, p_embeddings
    )  # (n_queries, n_passages)
    target = torch.arange(len(q_embeddings)).to(q_embeddings.device)  # (n_queries,)
    loss = F.cross_entropy(similarity, target)
    return loss, similarity


@torch.inference_mode()
def calc_relative_advantage(similarity):
    """
    similarity: (n_queries, n_passages)
    """
    sorted_similarity = torch.sort(similarity, 1, descending=True)[0]
    second_best = sorted_similarity[:, 1]
    return (torch.diag(similarity) - second_best) / second_best


class BM25:
    """
    bm25 = BM25(k1=0.9, b=0.4)
    bm.index(corpus)
    bm.retrieve(queries, top_k=100)
    """

    def __init__(self, pattern=r"(?u)\b\w\w+\b", k1=0.9, b=0.4):
        """
        The default `k1` and `b` are consistent with Karpukhin et al. (2020).
        """
        self.pattern = re.compile(pattern)
        self.k1 = k1
        self.b = b

    def split(self, text: str):
        text = text.lower()
        text = self.pattern.findall(text)
        return text

    def index(self, corpus: List[str]):
        self.chunk_lens = []
        self.freqs = []
        token_to_num_chunks = Counter()
        if isinstance(corpus, str):
            corpus = [corpus]
        for text in tqdm(corpus):
            tokens = self.split(text)
            chunk_len = len(tokens)
            freqs = {}
            for token in tokens:
                if token not in freqs:
                    freqs[token] = 1
                    token_to_num_chunks[token] += 1
                else:
                    freqs[token] += 1
            self.chunk_lens.append(chunk_len)
            self.freqs.append(freqs)
        self.num_chunks = len(self.chunk_lens)
        self.mean_chunk_len = np.mean(self.chunk_lens).item()
        self.token_to_num_chunks = dict(token_to_num_chunks)

    def retrieve(self, queries: List[str], top_k=100):
        if isinstance(queries, str):
            queries = [queries]
        results = []
        for query in tqdm(queries):
            result = self._retrieve(query, top_k)
            results.append(result)
        indexes = np.stack([result[0] for result in results])
        scores = np.stack([result[1] for result in results])
        return indexes, scores

    def _retrieve(self, query: str, top_k=100):
        if top_k is None:
            top_k = self.num_chunks
        assert top_k <= self.num_chunks
        query_tokens = self.split(query)
        scores = np.zeros(self.num_chunks, dtype=float)
        for chunk_index in range(self.num_chunks):
            score = self.score(query_tokens, chunk_index)
            scores[chunk_index] = score
        indexes = np.argsort(-scores)[:top_k]
        scores = scores[indexes]
        return indexes, scores

    def score(self, query_tokens: List[str], chunk_index: int):
        chunk_len = self.chunk_lens[chunk_index]
        n = np.array(
            list(
                map(lambda token: self.token_to_num_chunks.get(token, 0), query_tokens)
            )
        )
        freqs = np.array(
            list(map(lambda token: self.freqs[chunk_index].get(token, 0), query_tokens))
        )
        IDF = self.calc_IDF(n)
        weight = freqs / (
            freqs + self.k1 * (1 - self.b + self.b * chunk_len / self.mean_chunk_len)
        )
        return np.sum(IDF * weight)

    def calc_IDF(self, n: int):
        """
        n: number of chunks containing the term
        """
        return np.log((self.num_chunks - n + 0.5) / (n + 0.5) + 1)


# ----- train -----
@dataclass
class TrainArgs:
    save_dir: Union[str, Path] = project_dir / "checkpoints"  # 模型保存目录
    save_weight: str = "dpr"  # 保存权重的前缀名
    epochs: int = 1  # 训练轮数（建议1轮zero或2-6轮充分训练
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    device: str = None
    dtype: str = "bfloat16"
    num_workers: int = 0  # 数据加载线程数
    accumulation_steps: int = 2  # 梯度累积步数
    grad_clip: float = 1.0  # 梯度裁剪阈值
    log_interval: int = 100  # 日志打印间隔
    save_interval: int = 100  # 模型保存间隔
    hidden_size: int = 512  # 隐藏层维度
    num_hidden_layers: int = 8  # 隐藏层数量
    max_query_len: int = 64
    max_passage_len: int = 512
    use_amp: bool = None  # 使用自动混合精度
    temperature: float = 1.0
    data_path: Union[str, Path] = (
        project_dir / "dataset/huatuo_encyclopedia_qa_512/train.jsonl"
    )
    from_weight: str = "pretrain"  # 基于哪个权重训练，为none则从头开始
    from_resume: int = 0  # 是否自动检测&续训（0=否，1=是）
    use_wandb: bool = False
    use_swanlab: bool = False
    use_bert: bool = (False,)
    wandb_entity: str = None
    wandb_project: str = "MiniMind"
    profile: bool = False

    def __post_init__(self):
        assert self.dtype in ["bfloat16", "float16"]
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.use_amp is None:
            self.use_amp = "cuda" in self.device
        self.save_dir = Path(self.save_dir).as_posix()
        self.data_path = Path(self.data_path)
        self.dataset_name = self.data_path.name
        self.data_path = self.data_path.as_posix()
        if self.profile:
            self.use_wandb = False
        self.wandb = wandb
        if self.use_swanlab:
            import swanlab

            self.wandb = swanlab


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(
    model: MiniMindForCausalLM,
    args: TrainArgs,
    lm_config: MiniMindConfig,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    scaler: torch.GradScaler,
    epoch: int,
    loader: DataLoader,
    iters: int,
    bar,
    start_step=0,
    use_wandb=False,
    spend_time=0,
):
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    start_time = time.time()
    grad_norm = float("nan")
    wandb = args.wandb

    def get_spend_time():
        return spend_time + time.time() - start_time

    for step, (query_inputs, passage_inputs) in enumerate(loader, start=start_step + 1):
        query_inputs = query_inputs.to(args.device)
        passage_inputs = passage_inputs.to(args.device)
        if args.profile:
            print("data to deivce:", get_spend_time())

        with torch.autocast(device_type=args.device, enabled=args.use_amp, dtype=dtype):
            loss, similarity = model(query_inputs, passage_inputs)
            if args.profile:
                torch.cuda.synchronize()
                print("forward:", get_spend_time())
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()
        if args.profile:
            print("backward:", get_spend_time())
        loss = loss.detach().cpu().numpy().item() * args.accumulation_steps
        lr = optimizer.param_groups[-1]["lr"]

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            grad_norm = grad_norm.detach().cpu().numpy().item()

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()
            torch.cuda.empty_cache()

            if use_wandb:
                adv = (
                    calc_relative_advantage(similarity)
                    .mean()
                    .detach()
                    .cpu()
                    .numpy()
                    .item()
                )
                wandb.log(
                    {
                        "train_step": step // args.accumulation_steps,
                        "train/epoch": epoch + 1,
                        "train/loss": loss,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
                        "train/adv": adv,
                        "train/time": get_spend_time(),
                    }
                )
        scheduler.step()
        if bar is not None:
            bar.update()
            bar.set_postfix_str(
                f"[{epoch+1}/{args.epochs}]loss={round(loss, 4)},grad_norm={round(grad_norm, 4)}"
            )

        if step % args.log_interval == 0 or step == iters - 1:
            Logger(f"Epoch:[{epoch+1}/{args.epochs},{step}/{iters}] loss:{loss:.4f}")

        if args.profile:
            return
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}.pth"
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            wandb_id = None
            if args.use_wandb:
                wandb_id = getattr(wandb.run, "id", None)
            lm_checkpoint(
                lm_config,
                weight=args.save_weight,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                epoch=epoch,
                step=step,
                wandb_id=wandb_id,
                save_dir=args.save_dir,
                spend_time=get_spend_time(),
            )
            model.train()
    return get_spend_time()


def train(
    args: TrainArgs,
    model: MiniMindForCausalLM = None,
    tokenizer: PreTrainedTokenizerBase = None,
    train_ds: DPRDataset = None,
    early_return=False,
):
    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=False,
    )
    ckp_data = (
        lm_checkpoint(lm_config, weight=args.save_weight, save_dir=args.save_dir)
        if args.from_resume == 1
        else None
    )

    # ========== 3. 设置混合精度 ==========
    # set in TrainArgs

    # ========== 4. 定义模型、数据、优化器 ==========
    if model is None:
        model, tokenizer = init_model(
            lm_config,
            args.from_weight,
            tokenizer_path=project_dir / "model",
            save_dir=args.save_dir,
            device=args.device,
        )
        model = Encoder(model, temperature=args.temperature)
    num_params = get_num_params(model)
    if train_ds is None:
        train_ds = load_nq_dataset(args.max_query_len, args.max_passage_len)
        train_ds = DPRDataset(
            train_ds["train"],
            tokenizer,
            max_query_len=args.max_query_len,
            max_passage_len=args.max_passage_len,
        )

    if early_return:  # for debugging
        return model, tokenizer, train_ds

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.GradScaler(args.device, enabled=(args.dtype == "float16"))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=args.betas,
    )

    # ========== 5. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    spend_time = 0
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])
        scaler.load_state_dict(ckp_data["scaler"])
        start_epoch = ckp_data["epoch"]
        start_step = ckp_data.get("step", 0)
        spend_time = ckp_data.get("spend_time", 0)

    if start_step > 0:  # 第一个epoch且存在检查点
        batch_sampler = SkipBatchSampler(
            train_sampler or range(len(train_ds)), args.batch_size, start_step + 1
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:  # 默认从头开始
        loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    iters = len(loader)
    max_step = args.epochs * iters
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_step)
    if start_step > 0:
        scheduler.load_state_dict(ckp_data["scheduler"])

    # ========== 6. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 7. 配wandb ==========
    run = nullcontext()
    use_wandb = args.use_wandb and is_main_process()
    if use_wandb:
        wandb = args.wandb
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"MiniMind-{round(num_params/1e6)}M-{args.save_weight}-epoch-{args.epochs}-batchsize-{args.batch_size}-lr-{args.learning_rate}"
        if wandb_id:
            config = None
        else:
            config = asdict(args)
            for key in [
                "device",
                "from_resume",
                "use_wandb",
                "wandb_project",
                "save_dir",
                "data_path",
            ]:
                config.pop(key)
            config["max_step"] = max_step
            config["num_params"] = num_params
            config["dataset_name"] = args.dataset_name
        wandb_kwargs = dict(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=wandb_run_name,
            id=wandb_id,
            resume=resume,
            config=config,
        )
        if args.use_swanlab:
            wandb_kwargs.pop("entity")
        run = wandb.init(**wandb_kwargs)
        if not args.use_swanlab:
            wandb.define_metric("train_step")
            wandb.define_metric("eval_step")
            wandb.define_metric("train/*", step_metric="train_step")
            wandb.define_metric("eval/*", step_metric="eval_step")

    # ========== 8. 开始训练 ==========
    bar = tqdm(total=max_step, disable=not is_main_process())
    with run:
        for epoch in range(start_epoch, args.epochs):
            train_sampler and train_sampler.set_epoch(epoch)
            if epoch == start_epoch and start_step > 0:  # 第一个epoch且存在检查点
                Logger(
                    f"Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始"
                )
                # train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
                _iters = iters + start_step + 1
                bar.update(start_step)
            else:  # 默认从头开始
                _iters = iters
            spend_time = train_epoch(
                model=model,
                args=args,
                lm_config=lm_config,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                loader=loader,
                iters=_iters,
                start_step=start_step,
                bar=bar,
                use_wandb=use_wandb,
                spend_time=spend_time,
            )
            if args.profile:
                return
    bar.close()
