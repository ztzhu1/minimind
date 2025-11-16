from pathlib import Path
import sys

project_dir = Path(__file__).parent.parent
project_path = project_dir.as_posix()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import os
import time
from typing import Union
import warnings

import torch
from torch import nn, optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import wandb

from dataset.lm_dataset import SFTDataset
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


@dataclass
class TrainArgs:
    save_dir: Union[str, Path] = project_dir / "checkpoints"  # 模型保存目录
    save_weight: str = "sft"  # 保存权重的前缀名
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    device: str = None
    dtype: str = "bfloat16"
    num_workers: int = 0  # 数据加载线程数
    accumulation_steps: int = 8  # 梯度累积步数
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
