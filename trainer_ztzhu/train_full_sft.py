from pathlib import Path
import sys

project_dir = Path(__file__).parent.parent
project_path = project_dir.as_posix()
if project_path not in sys.path:
    sys.path.insert(0, project_path)
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import os
import time
from typing import Union
import warnings

import datasets
from openai import OpenAI
import pandas as pd
import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
import wandb

from model.model_minimind import MiniMindConfig
from trainer_ztzhu.trainer_utils import (
    Logger,
    MiniMindForCausalLM,
    SkipBatchSampler,
    get_lr,
    get_num_params,
    init_distributed_mode,
    init_model,
    is_main_process,
    lm_checkpoint,
    setup_seed,
)

warnings.filterwarnings("ignore")

api_keys = None


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


def make_sft_dataset(dataset, batch_size=1, indexes=None, counterexample=False):
    global api_keys
    if api_keys is None:
        with open(project_dir / "api_keys.json", "r") as f:
            api_keys = json.load(f)

    path = project_dir.joinpath("dataset", "ChineseSquad_train.jsonl")
    sft_data = load_jsonl(path)

    with open(project_dir / "dataset" / "annot_ChineseSquad.prompt", "r") as f:
        pre_prompt = f.read()

    if len(sft_data) == 0:
        sft_data = pd.DataFrame(columns=["question", "context", "answers", "answer"])
        start = 0
    else:
        start = len(sft_data)
    if counterexample:
        offset = 200
    else:
        offset = 0

    # client = OpenAI(api_key=api_keys["deepseek"], base_url="https://api.deepseek.com")
    client = OpenAI(
        api_key=api_keys["chatanywhere"], base_url="https://api.chatanywhere.tech"
    )

    def get_response(index):
        if counterexample:
            sample = deepcopy(dataset[index])
            sample["question"] = dataset[index + offset]["question"]
            content = pre_prompt + str(sample)
        else:
            content = pre_prompt + str(dataset[index])
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a dataset annotation expert.",
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            stream=False,
        )
        response = response.choices[0].message.content
        return index, response

    pool = ThreadPoolExecutor()
    results = []
    if indexes is None:
        indexes = range(start, start + batch_size)
    for index in indexes:
        results.append(pool.submit(get_response, index))

    for result in results:
        index, response = result.result()
        sft_data.loc[index] = {
            "question": dataset[index + offset]["question"],
            "context": dataset[index]["context"],
            "answers": dataset[index]["answers"],
            "answer": response,
        }
    save_jsonl(sft_data, path, overwrite=True)


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.jsonl_path = jsonl_path
        self.samples = load_jsonl(jsonl_path)
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples.iloc[index]
        prompt = f'你是一个智能问答助手，请严格按照以下要求回答问题： 如果资料中包含问题答案，请直接使用资料信息并注明"根据资料"，如果资料不相关、信息不足或未包含答案，请明确说明"资料中未包含相关信息"，然后可以基于常识进行补充\n资料：{sample["context"]}\n问题：{sample["question"]}\n现在请开始回答：'
        prompt = f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        response = f"{sample['answer']}<|im_end|>"
        result = self.tokenizer(
            [(prompt, response)],
            padding="max_length",
            padding_side="right",
            return_token_type_ids=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = result.input_ids.view(-1)
        loss_mask = result.token_type_ids.view(-1)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask


@dataclass
class TrainArgs:
    save_dir: Union[str, Path] = project_dir / "checkpoints"  # 模型保存目录
    save_weight: str = "sft"  # 保存权重的前缀名
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 2e-6
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
    use_moe: int = 0  # 是否使用MoE架构（0=否，1=是）
    use_amp: bool = None  # 使用自动混合精度
    data_path: Union[str, Path] = project_dir / "dataset/sft_mini_512.jsonl"  # 预训练数据路径
    model_dir = project_dir / "out"
    from_weight: str = "none"  # 基于哪个权重训练，为none则从头开始
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
        self.model_dir = Path(self.model_dir)
        if not self.model_dir.exists():
            new_model_dir = self.model_dir.parent / "checkpoints"
            print(
                f"model dir {self.model_dir.as_posix()} not exists, switch to {new_model_dir.as_posix()}"
            )
            self.model_dir = new_model_dir
        self.model_dir = self.model_dir.as_posix()
        if self.profile:
            self.use_wandb = False
        self.wandb = wandb
        if self.use_swanlab:
            import swanlab

            self.wandb = swanlab


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
    grad_norm = float("nan")
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    wandb = args.wandb
    start_time = time.time()

    def get_spend_time():
        return spend_time + time.time() - start_time

    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        with torch.autocast(device_type=args.device, enabled=args.use_amp, dtype=dtype):
            res = model(X)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(
                Y.size()
            )

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()
        loss = loss.detach().cpu().numpy().item() * args.accumulation_steps
        lr = optimizer.param_groups[-1]["lr"]

        if (step + 1) % args.accumulation_steps == 0:
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
                        "train_step": step // args.accumulation_steps,
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
                f"[{epoch+1}/{args.epochs}]loss={loss:.4f},grad_norm={grad_norm:.4f}"
            )

        if step % args.log_interval == 0 or step == iters - 1:
            Logger(f"Epoch:[{epoch+1}/{args.epochs},{step}/{iters}] loss:{loss:.4f}")

        if args.profile:
            return
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth"
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
    tokenizer=None,
    train_ds: SFTDataset = None,
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
        use_moe=bool(args.use_moe),
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
            save_dir=args.model_dir,
            device=args.device,
        )
    num_params = get_num_params(model)
    if train_ds is None:
        train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    if early_return:
        return model, tokenizer, train_ds
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.GradScaler(enabled=(args.dtype == "float16"))
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
                "model_dir",
            ]:
                config.pop(key, None)
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
