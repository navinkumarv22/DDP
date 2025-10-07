# train_gpt_nano.py
# -----------------------------------------------------------------------------
# NanoGPT-style minimal GPT trainer over token shards, with --impl torch|custom
# Default data root points to your path:
#   /home/blubridge-035/Desktop/GPT_by_Navin/edu_fineweb10B
# -----------------------------------------------------------------------------

import os, math, time, inspect
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

# ---------------------------- Model ------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        # Flash attention path (PyTorch 2.x)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc   = nn.Linear(n_embd, 4 * n_embd)
        self.gelu   = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config.n_embd, config.n_head) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() <  2]
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {sum(p.numel() for p in decay_params):,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {sum(p.numel() for p in nodecay_params):,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

# ------------------------- Data Loader (shards) -------------------------------

def load_tokens(filename: str) -> torch.Tensor:
    npt = np.load(filename)
    npt = np.asarray(npt, dtype=np.int32)
    return torch.from_numpy(npt.astype(np.int64))  # torch.long

class DataLoaderLite:
    """
    NanoGPT-style token loader over .npy shards. Filenames should include 'train' or 'val'.
    Each shard is a 1D array of token ids. Produces (x, y) of shape (B, T).
    """
    def __init__(self, B, T, process_rank, num_processes, split, data_root):
        assert split in {"train", "val"}
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        shards = sorted(
            os.path.join(data_root, s)
            for s in os.listdir(data_root)
            if split in s and s.endswith(".npy")
        )
        assert shards, f"no shards found for split '{split}' in {data_root}"
        self.shards = shards
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def _advance_shard(self):
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        end = self.current_position + B * T + 1
        if end > len(self.tokens):
            self._advance_shard()
            end = self.current_position + B * T + 1
        buf = self.tokens[self.current_position : end]
        x = buf[:-1].view(B, T)
        y = buf[1: ].view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self._advance_shard()
        return x, y

# ----------------------------- Config ----------------------------------------

@dataclass
class TrainConfig:
    impl: str = "custom"  # "custom" or "torch"
    data_root: str = "/home/blubridge-035/Desktop/GPT_by_Navin/edu_fineweb10B"

    # model
    block_size: int = 256
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

    # global batch (tokens) = (B*T*world) * grad_accum_steps
    B: int = 4
    T: int = 256
    total_batch_tokens: int = 524288  # ~0.5M tokens per optimizer step

    max_steps: int = 200
    weight_decay: float = 0.1
    max_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 10

    eval_every: int = 50
    eval_steps: int = 20

    bucket_mb: int = 25
    log_dir: str = "log_nano"
    seed: int = 1337

# --------------------------- Schedules/Utils ---------------------------------

def build_scheduler(cfg: TrainConfig):
    max_lr = cfg.max_lr
    min_lr = cfg.max_lr * cfg.min_lr_ratio
    warmup = cfg.warmup_steps
    total  = cfg.max_steps
    def get_lr(step):
        if step < warmup:
            return max_lr * (step + 1) / max(warmup, 1)
        if step >= total:
            return min_lr
        ratio = (step - warmup) / max(total - warmup, 1)
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return min_lr + coeff * (max_lr - min_lr)
    return get_lr

def init_distributed():
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        ddp_rank        = int(os.environ['RANK'])
        ddp_local_rank  = int(os.environ['LOCAL_RANK'])
        ddp_world_size  = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(ddp_local_rank)
        master = (ddp_rank == 0)
    else:
        ddp_rank = 0; ddp_local_rank = 0; ddp_world_size = 1; master = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if master: print(f"using device: {device}")
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master, device

# ------------------------------- Main ----------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["custom", "torch"], default="custom")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()

    cfg = TrainConfig(impl=args.impl)
    if args.data_root: cfg.data_root = args.data_root
    if args.max_steps is not None: cfg.max_steps = args.max_steps

    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = init_distributed()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    torch.manual_seed(cfg.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.set_float32_matmul_precision('high')

    # data
    train_loader = DataLoaderLite(B=cfg.B, T=cfg.T, process_rank=ddp_rank, num_processes=ddp_world_size,
                                  split="train", data_root=cfg.data_root)
    val_loader   = DataLoaderLite(B=cfg.B, T=cfg.T, process_rank=ddp_rank, num_processes=ddp_world_size,
                                  split="val",   data_root=cfg.data_root)

    # model
    gcfg = GPTConfig(block_size=cfg.block_size, vocab_size=cfg.vocab_size,
                     n_layer=cfg.n_layer, n_head=cfg.n_head, n_embd=cfg.n_embd)
    model = GPT(gcfg).to(device)

    # impl switch
    if cfg.impl == "torch":
        if ddp:
            dist.init_process_group(backend="nccl")
            from torch.nn.parallel import DistributedDataParallel as TorchDDP
            model = TorchDDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank,
                             broadcast_buffers=False, bucket_cap_mb=cfg.bucket_mb)
        raw_model = model.module if ddp else model
    else:
        if ddp:
            from ddp.process_group import init_process_group as init_pg_custom
            from ddp.ddp import DistributedDataParallel as CustomDDP, ParamSpec
            pg, _, _, _ = init_pg_custom()
            specs = [ParamSpec(p.numel(), ddp_local_rank, p) for p in model.parameters()]
            ddp_wrap = CustomDDP(specs, pg, bucket_bytes=cfg.bucket_mb * 1024 * 1024)
            if master_process: print("Broadcasting parameters...")
            ddp_wrap.broadcast_parameters(root=0)
        raw_model = model  # forward uses raw model
    optimizer = raw_model.configure_optimizers(cfg.weight_decay, cfg.max_lr, device_type)
    get_lr = build_scheduler(cfg)

    # grad accumulation math (GLOBAL tokens per step)
    assert cfg.total_batch_tokens % (cfg.B * cfg.T * ddp_world_size) == 0, \
        "total_batch_tokens must be divisible by B*T*world_size"
    grad_accum_steps = cfg.total_batch_tokens // (cfg.B * cfg.T * ddp_world_size)
    if master_process:
        print(f"total desired batch tokens: {cfg.total_batch_tokens}")
        print(f"=> grad_accum_steps: {grad_accum_steps} (B={cfg.B}, T={cfg.T}, world={ddp_world_size})")

    # logging setup
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = os.path.join(cfg.log_dir, "log.txt")
    with open(log_file, "w"): pass

    # training loop
    for step in range(cfg.max_steps):
        t0 = time.time()
        last_step = (step == cfg.max_steps - 1)

        # LR
        lr = get_lr(step)
        for pg in optimizer.param_groups: pg['lr'] = lr

        # ---- validation (periodic) ----
        if (step % cfg.eval_every == 0) or last_step:
            raw_model.eval(); val_loader.reset()
            with torch.no_grad():
                val_loss_accum = torch.tensor(0.0, device=device)
                for _ in range(cfg.eval_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type=="cuda" else torch.float32):
                        _, loss = raw_model(x, y)
                    val_loss_accum += loss
                val_loss_accum /= cfg.eval_steps
            if ddp and cfg.impl == "torch":
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"val loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        # ---- train step ----
        raw_model.train()
        optimizer.zero_grad(set_to_none=True)
        train_loader.reset()
        loss_accum = torch.tensor(0.0, device=device)

        for micro in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if ddp and cfg.impl == "torch":
                # only sync on last micro step (classic DDP trick)
                model.require_backward_grad_sync = (micro == grad_accum_steps - 1)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type=="cuda" else torch.float32):
                _, loss = raw_model(x, y)

            (loss / grad_accum_steps).backward()
            loss_accum += loss.detach()

            if ddp and cfg.impl == "custom":
                ddp_wrap.synchronize()  # run allreduce/avg after grads land in buckets

        # reduce loss metric for fair logging
        if ddp and cfg.impl == "torch":
            la = loss_accum.clone()
            dist.all_reduce(la, op=dist.ReduceOp.AVG)
            loss_print = la.item()
        else:
            loss_print = loss_accum.item()

        norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()

        if device_type == "cuda": torch.cuda.synchronize()
        dt = time.time() - t0
        toks = cfg.B * cfg.T * grad_accum_steps * ddp_world_size
        tps = toks / dt

        if master_process:
            print(f"step {step:5d} | loss: {loss_print:.6f} | lr {lr:.2e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/s: {tps:.0f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_print:.6f}\n")

    if ddp and cfg.impl == "torch":
        dist.destroy_process_group()
