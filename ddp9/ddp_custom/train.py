# train_mlp.py
# -----------------------------------------------------------------------------

import os, math, time, inspect, glob, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F


# ---- Graph export helpers (CPU-only, robust) -------------------------------
# def dump_graphs_cpu(model_raw: torch.nn.Module, cfg, is_rank0: bool):
#     """
#     Writes:
#       outputs/autograd_graph.dot  (always)
#       outputs/autograd_graph.pdf  (if graphviz 'dot' is installed)
#       outputs/model.onnx
#     All exports run on CPU to avoid device-mismatch and FakeTensor issues.
#     """
#     if not is_rank0:
#         return
#     os.makedirs("outputs", exist_ok=True)

#     # Build tiny CPU batch that matches TinyLM.forward signature
#     B = min(2, cfg.micro_batch_size if hasattr(cfg, "micro_batch_size") else 2)
#     T = min(16, cfg.seq_len)
#     idx_cpu = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)

#     # --- torchviz autograd graph (forward+loss on CPU) ---
#     try:
#         from torchviz import make_dot  # pip install torchviz
#         model_cpu = model_raw.to("cpu").eval()
#         logits = model_cpu(idx_cpu)  # (B,T,V)
#         tgt_cpu = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)
#         loss = torch.nn.functional.cross_entropy(
#             logits.view(-1, cfg.vocab_size), tgt_cpu.view(-1)
#         )
#         dot = make_dot(loss, params=dict(model_cpu.named_parameters()))
#         dot.save("outputs/autograd_graph.dot")
#         print("[rank0] wrote outputs/autograd_graph.dot")
#         try:
#             dot.render("outputs/autograd_graph", format="pdf")
#             print("[rank0] wrote outputs/autograd_graph.pdf")
#         except Exception as e:
#             print("[rank0] graphviz 'dot' not found; install it for PDF:", e)
#     except Exception as e:
#         print("[rank0] torchviz export failed:", e)

#     # --- ONNX (CPU) ---
#     try:
#         model_onnx = model_raw.to("cpu").eval()
#         dummy = torch.randint(0, cfg.vocab_size, (B, T), dtype=torch.long)
#         torch.onnx.export(
#             model_onnx, dummy, "outputs/model.onnx",
#             input_names=["input_ids"], output_names=["logits"],
#             opset_version=17, do_constant_folding=True, dynamic_axes=None,
#             # legacy exporter is OK here; avoid torch.export for embeddings
#             dynamo=False
#         )
#         print("[rank0] wrote outputs/model.onnx  (open with Netron)")
#     except Exception as e:
#         print("[rank0] ONNX export failed:", e)



# ----------------------------- Your MLP block --------------------------------

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# --------------------------- Tiny LM wrapper ---------------------------------

class TinyLMConfig:
    def __init__(self, n_embd: int, vocab_size: int, block_size: int):
        self.n_embd = n_embd
        self.vocab_size = vocab_size
        self.block_size = block_size

class TinyLM(nn.Module):
    """
    Token + Pos embedding -> Your MLP (position-wise) -> LM head to vocab.
    Note: No attention; objective is next-token prediction to validate DDP.
    """
    def __init__(self, cfg: TinyLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.mlp     = MLP(cfg)
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # (Optional) init similar to NanoGPT for projection
        with torch.no_grad():
            self.lm_head.weight.mul_(0.5)

    def forward(self, idx):  # idx: (B, T) int64
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence length exceeds block_size"
        tok = self.tok_emb(idx)                                  # (B,T,C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))   # (T,C)
        x = tok + pos.unsqueeze(0)                               # (B,T,C)
        x = self.mlp(x)                                          # (B,T,C)
        x = self.ln_f(x)                                         # (B,T,C)
        logits = self.lm_head(x)                                 # (B,T,V)
        return logits

# ------------------------------- Config --------------------------------------

@dataclass
class TrainConfig:
    impl: str = "custom"         # "custom" or "torch"
    n_embd: int = 256
    vocab_size: int = 50304      # NanoGPT default pad to 64
    seq_len: int = 512           # T

    # global batch config (samples per optimizer step across all ranks)
    total_batch_size: int = 65536
    micro_batch_size: int = 16

    max_steps: int = 306
    weight_decay: float = 0.1
    max_lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 10

    seed: int = 1337
    log_dir: str = "log_mlp"
    bucket_mb: int = 25
    
    tokens_per_shard: Optional[int] = None
    shard_offset: int = 0

    # data
    data_root: str = "/home/blubridge-035/Desktop/GPT_by_Navin/edu_fineweb10B"

# ------------------------------- Scheduler -----------------------------------

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

# ------------------------------ DDP bootstrap --------------------------------

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
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master = True
        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        if master:
            print(f"using device: {device}")
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master, device

# --------------------------- NumPy token loader -------------------------------

N_TOKENS_TO_LOAD = 10000000   # e.g. 10_000_000 (None = full shard)
TOKEN_OFFSET     = 0      # starting index within each shard

def load_tokens(filename):
    npt = np.load(filename, mmap_mode="r")
    print(npt.dtype)
    npt = npt.reshape(-1)
    start = TOKEN_OFFSET if TOKEN_OFFSET is not None else 0
    end   = None if N_TOKENS_TO_LOAD is None else start + int(N_TOKENS_TO_LOAD)
    npt = npt[start:end]
    npt = npt.astype(np.int32, copy=False)   # added after video
    ptt = torch.from_numpy(npt.astype(np.int64, copy=False))  # torch.long
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split,
                 data_root="edu_fineweb10B",
                 tokens_per_shard=None,
                 shard_offset=0):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        self.data_root = data_root
        self.tokens_per_shard = tokens_per_shard
        self.shard_offset = shard_offset

        # get the shard filenames
        shards = os.listdir(self.data_root)
        shards = [s for s in shards if split in s and s.endswith(".npy")]
        shards = sorted(shards)
        shards = [os.path.join(self.data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split} in {self.data_root}"
        if 'master_process' in globals() and master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def _set_limits(self):
        # keep your load_tokens() API by setting globals before each load
        global N_TOKENS_TO_LOAD, TOKEN_OFFSET
        N_TOKENS_TO_LOAD = self.tokens_per_shard
        TOKEN_OFFSET     = self.shard_offset

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self._set_limits()
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if the next read would go out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self._set_limits()
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# --------------------------------- Main --------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["custom", "torch"], default="custom")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--vocab_size", type=int, default=None)
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--bucket_mb", type=int, default=None)
    ap.add_argument("--tokens_per_shard", type=int, default=None)
    ap.add_argument("--shard_offset", type=int, default=0)

    args = ap.parse_args()

    cfg = TrainConfig(impl=args.impl)
    if args.max_steps is not None:   cfg.max_steps  = args.max_steps
    if args.data_root is not None:   cfg.data_root  = args.data_root
    if args.seq_len is not None:     cfg.seq_len    = args.seq_len
    if args.vocab_size is not None:  cfg.vocab_size = args.vocab_size
    if args.n_embd is not None:      cfg.n_embd     = args.n_embd
    if args.bucket_mb is not None:         cfg.bucket_mb = args.bucket_mb
    if args.tokens_per_shard is not None:  cfg.tokens_per_shard = args.tokens_per_shard
    if args.shard_offset is not None:      cfg.shard_offset = args.shard_offset

    # DDP env
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = init_distributed()
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    torch.manual_seed(cfg.seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.set_float32_matmul_precision('high')

    # model
    model_cfg = TinyLMConfig(n_embd=cfg.n_embd, vocab_size=cfg.vocab_size, block_size=cfg.seq_len)
    model = TinyLM(model_cfg).to(device)

    # model_raw = model  # keep handle pre-DDP
    # is_rank0 = int(os.environ.get("RANK", "0")) == 0
    # dump_graphs_cpu(model_raw, cfg, is_rank0)
    # impl switch
    ddp_wrap = None
    if cfg.impl == "torch":
        if ddp:
            dist.init_process_group(backend="nccl")
            from torch.nn.parallel import DistributedDataParallel as TorchDDP
            model = TorchDDP(model, device_ids=[ddp_local_rank], output_device=ddp_local_rank,
                             broadcast_buffers=False, bucket_cap_mb=cfg.bucket_mb)
    else:
        if ddp:
            from ddp.process_group import init_process_group as init_pg_custom
            from ddp.ddp import DistributedDataParallel as CustomDDP, ParamSpec
            pg, _, _, _ = init_pg_custom()
            specs = [ParamSpec(p.numel(), ddp_local_rank, p) for p in model.parameters()]
            ddp_wrap = CustomDDP(specs, pg, bucket_bytes=cfg.bucket_mb * 1024 * 1024)
            if master_process:
                print("Broadcasting parameters...")
            ddp_wrap.broadcast_parameters(root=0)

    # optimizer + scheduler
    opt_target = model.module if (cfg.impl == "torch" and ddp) else model
    # NanoGPT-style WD groups (2D get WD, 1D no WD)
    param_dict = {pn: p for pn, p in opt_target.named_parameters() if p.requires_grad}
    decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() <  2]
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': cfg.weight_decay},
         {'params': nodecay_params, 'weight_decay': 0.0}],
        lr=cfg.max_lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
    )
    get_lr = build_scheduler(cfg)

    # grad accumulation math
    assert cfg.total_batch_size % (cfg.micro_batch_size * cfg.seq_len * ddp_world_size) == 0, \
        "total_batch_size must be divisible by micro_batch_size * world_size"
    grad_accum_steps = cfg.total_batch_size // (cfg.micro_batch_size * cfg.seq_len * ddp_world_size)
    if master_process:
        print(f"total desired global batch: {cfg.total_batch_size}")
        print(f"=> grad_accum_steps: {grad_accum_steps} (micro {cfg.micro_batch_size}, world {ddp_world_size})")

    # logging
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = os.path.join(cfg.log_dir, "log.txt")
    with open(log_file, "w"): pass

    # data
    B, T = cfg.micro_batch_size, cfg.seq_len
    train_loader = DataLoaderLite(
    B=B, T=T,
    process_rank=ddp_rank, num_processes=ddp_world_size,
    split="train", data_root=cfg.data_root,
    tokens_per_shard=cfg.tokens_per_shard,
    shard_offset=cfg.shard_offset,
    )
    val_loader = DataLoaderLite(
        B=B, T=T,
        process_rank=ddp_rank, num_processes=ddp_world_size,
        split="val", data_root=cfg.data_root,
        tokens_per_shard=cfg.tokens_per_shard,
        shard_offset=cfg.shard_offset,
    )


    # training loop (next-token CE)
    for step in range(cfg.max_steps):
        t0 = time.time()

        # LR schedule
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for micro in range(grad_accum_steps):
            x_int, y_int = train_loader.next_batch()  # (B,T) int64
            x = x_int.to(device, non_blocking=True)
            y = y_int.to(device, non_blocking=True)

            if ddp and cfg.impl == "torch":
                model.require_backward_grad_sync = (micro == grad_accum_steps - 1)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16 if device_type=="cuda" else torch.float32):
                logits = model(x)  # (B,T,V)
                loss = F.cross_entropy(
                    logits.view(-1, cfg.vocab_size),
                    y.view(-1)
                )

            (loss / grad_accum_steps).backward()
            loss_accum += float(loss.detach())

        if ddp and cfg.impl == "custom":
            ddp_wrap.synchronize()

        # (optional) world-avg metric for torch DDP
        loss_acc_num = loss_accum
        if ddp and cfg.impl == "torch":
            la = torch.tensor([loss_acc_num], device=device, dtype=torch.float32)
            dist.all_reduce(la, op=dist.ReduceOp.AVG)
            loss_acc_num = la.item()

        # clip + step
        norm = torch.nn.utils.clip_grad_norm_(opt_target.parameters(), 1.0)
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        if master_process:
            samples = cfg.micro_batch_size * grad_accum_steps * ddp_world_size * cfg.seq_len
            tok_per_s = samples / dt
            print(f"step {step:5d} | loss: {loss_acc_num:.6f} | lr {lr:.2e} | norm: {float(norm):.4f} | dt: {dt*1000:.2f}ms | tok/s: {tok_per_s:.0f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_acc_num:.6f}\n")

    if ddp and cfg.impl == "torch":
        dist.destroy_process_group()
