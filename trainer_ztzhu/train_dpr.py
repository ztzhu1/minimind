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
from typing import Union
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


def make_huatuo_512(dataset, key="test", overwrite=False):
    """
    dataset = datasets.load_dataset("FreedomIntelligence/huatuo_encyclopedia_qa")
    """
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


def load_dataset(tokenizer: PreTrainedTokenizerBase, ds_type="test"):
    assert ds_type in ["train", "test", "validation"]
    path = project_dir / "dataset" / "huatuo_encyclopedia_qa_512"
    dataset = DPRDataset(path / f"{ds_type}.jsonl", tokenizer)
    return dataset


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

        def encode(key):
            inputs = self.tokenizer(
                sample[key],
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
                padding_side="right",
                padding="max_length",
                return_token_type_ids=False,
            )
            inputs["input_ids"] = inputs["input_ids"].squeeze(0)
            inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
            return inputs

        query_inputs = encode("question")
        passage_inputs = encode("answer")
        return query_inputs, passage_inputs


# ----- model -----
class Retriever(nn.Module):
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
            attention_mask + torch.arange(attention_mask.shape[1]).to(attention_mask.device),
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
    max_seq_len: int = 512  # 训练的最大截断长度
    use_amp: bool = None  # 使用自动混合精度
    temperature: float = 1.0
    data_path: Union[str, Path] = (
        project_dir / "dataset/huatuo_encyclopedia_qa_512/train.jsonl"
    )
    from_weight: str = "pretrain"  # 基于哪个权重训练，为none则从头开始
    from_resume: int = 0  # 是否自动检测&续训（0=否，1=是）
    use_wandb: bool = False
    use_swanlab: bool = False
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

    def get_spend_time():
        return spend_time + time.time() - start_time

    for step, (query_inputs, passage_inputs) in enumerate(loader, start=start_step + 1):
        query_inputs = query_inputs.to(args.device)
        passage_inputs = passage_inputs.to(args.device)
        if args.profile:
            print("data to deivce:", get_spend_time())

        with torch.autocast(device_type=args.device, enabled=args.use_amp, dtype=dtype):
            loss = model(query_inputs, passage_inputs)[0]
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
                wandb.log(
                    {
                        "train_step": step // args.accumulation_steps + 1,
                        "train/epoch": epoch + 1,
                        "train/loss": loss,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm,
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
                if args.use_swanlab:
                    import swanlab as wandb
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
        model = Retriever(model, temperature=args.temperature)
    num_params = get_num_params(model)
    if train_ds is None:
        train_ds = DPRDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

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
        if args.use_swanlab:
            import swanlab as wandb
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
                "pre_trained_model_path",
            ]:
                config.pop(key)
            config["max_step"] = max_step
            config["num_params"] = num_params
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
