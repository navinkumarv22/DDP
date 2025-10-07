# ddp_compare.py
import os, time, math, argparse, csv
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F

# ---------- tiny MLP (same as before) ----------
class MLP(nn.Module):
    def __init__(self, dim=1024, hidden=4096):
        super().__init__()
        self.c_fc   = nn.Linear(dim, hidden)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(hidden, dim)
    def forward(self, x):  # (B, T, C) -> (B, T, C)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

def synthetic_batch(B=8, T=64, C=1024, device="cuda"):
    x = torch.randn(B, T, C, device=device)
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y

def setup_torch_pg():
    rank  = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group(backend="nccl")
    return rank, world, local

def cleanup_torch_pg():
    if dist.is_initialized():
        dist.destroy_process_group()

def grad_norm_l2(model: nn.Module, *, normalize_to_avg: bool, world_size: int) -> float:
    """Compute L2 norm of grads for logging. If normalize_to_avg=True, we scale
    grads by 1/world_size *before* computing the norm, so the reported value
    corresponds to 'averaged gradients' (matches your custom backend)."""
    total = 0.0
    scale = 1.0 / world_size if normalize_to_avg and world_size > 0 else 1.0
    for p in model.parameters():
        if p.grad is None: 
            continue
        g = p.grad.data.float()
        if scale != 1.0:
            g = g * scale
        n = g.norm(2).item()
        total += n * n
    return math.sqrt(total)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["torch", "custom"], required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--logdir", type=str, default="log_compare")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--bucket_mb", type=int, default=25)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.impl == "torch":
        rank, world, local = setup_torch_pg()
    else:
        # your custom PG + DDP
        from ddp.process_group import init_process_group
        from ddp.ddp import DistributedDataParallel, ParamSpec
        pg, rank, world, local = init_process_group()
        torch.cuda.set_device(local)

    device = f"cuda:{local}"
    model = MLP(dim=args.dim, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # ---------- wrap per-impl ----------
    if args.impl == "torch":
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        ddp_model = TorchDDP(
            model,
            device_ids=[local],
            output_device=local,
            broadcast_buffers=False,
            bucket_cap_mb=args.bucket_mb,     # align bucket size for fairness
        )
        wrapper = ddp_model
    else:
        # Create ParamSpecs for your custom DDP
        specs = []
        for p in model.parameters():
            assert p.is_cuda and p.device.index == local
            specs.append(ParamSpec(length_elems=p.numel(), device=local, p_ref=p))
        ddp_model = DistributedDataParallel(specs, pg, bucket_bytes=args.bucket_mb * 1024 * 1024)
        ddp_model.broadcast_parameters(root=0)
        wrapper = model  # forward uses raw model in the custom case

    # ---------- logging ----------
    os.makedirs(args.logdir, exist_ok=True)
    log_path = os.path.join(args.logdir, f"{args.impl}_rank{rank}.csv")
    if rank == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "grad_norm_avg", "elems_per_s", "ms_step"])

    B, T, C = args.batch, args.seq, args.dim

    for step in range(1, args.steps + 1):
        t0 = time.time()
        x, y = synthetic_batch(B, T, C, device=device)

        wrapper.train()
        opt.zero_grad(set_to_none=True)

        out = wrapper(x) if args.impl == "torch" else wrapper(x)
        loss = F.mse_loss(out, y)

        loss.backward()

        # Ensure reductions are complete before logging/stepping
        if args.impl == "custom":
            ddp_model.synchronize()                 # does allreduce + divide
            torch.cuda.synchronize()
            grad_norm_avg = grad_norm_l2(model, normalize_to_avg=False, world_size=world)  # already averaged
        else:
            torch.cuda.synchronize()                 # wait for DDP comm kernels
            # For LOGGING, normalize to averaged-grad convention for apples-to-apples
            grad_norm_avg = grad_norm_l2(model, normalize_to_avg=True, world_size=world)

        opt.step()

        torch.cuda.synchronize()
        dt = (time.time() - t0)
        elems = B * T * C
        tps = elems / dt

        # Average loss across ranks for logging fairness
        if args.impl == "torch":
            with torch.no_grad():
                lv = torch.tensor([loss.item()], device=device)
                dist.all_reduce(lv, op=dist.ReduceOp.AVG)
                loss_log = lv.item()
        else:
            loss_log = loss.item()

        if rank == 0:
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{loss_log:.6f}", f"{grad_norm_avg:.6f}", f"{tps:.2f}", f"{dt*1000:.2f}"])
            if step % 10 == 0 or step == args.steps:
                print(f"[{args.impl}] step {step:4d} | loss {loss_log:.6f} | grad(avg) {grad_norm_avg:.3f} | {tps:.0f} elems/s | {dt*1000:.1f} ms")

    if args.impl == "torch":
        cleanup_torch_pg()

if __name__ == "__main__":
    main()


# rm -rf build/ dist/ *.egg-info
# python3 -m pip install -v . -Ccmake.args=-DCMAKE_CUDA_ARCHITECTURES=86


# export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1

# torchrun --standalone --nproc_per_node=2 --master_port=29611 smoke_pg.py

# torchrun --standalone --nproc_per_node=2 --master_port=29611   compare.py --impl custom --steps 50
# torchrun --standalone --nproc_per_node=2 --master_port=29611   compare.py --impl torch --steps 50