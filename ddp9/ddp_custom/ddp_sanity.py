import os, sys, torch
from ddp.process_group import init_process_group

def abort(msg: str, code: int = 1):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(code)

def main():
    rank        = int(os.environ.get("RANK", "-1"))
    local_rank  = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size  = int(os.environ.get("WORLD_SIZE", "-1"))
    cvd         = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")

    print(f"[env] RANK={rank} LOCAL_RANK={local_rank} WORLD_SIZE={world_size} CVD={cvd}")

    if not torch.cuda.is_available():
        abort("CUDA not available. Check your drivers / device.")

    dev_count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(dev_count)]
    print(f"[cuda] visible device_count={dev_count} -> {names}")

    if local_rank < 0:
        abort("LOCAL_RANK not set. Use torchrun or set envs manually.")

    if local_rank >= dev_count:
        abort(f"LOCAL_RANK={local_rank} but only {dev_count} CUDA devices are visible. "
              f"Fix CUDA_VISIBLE_DEVICES or reduce --nproc_per_node to <= {dev_count}.")


    torch.cuda.set_device(local_rank)
    print(f"[cuda] using device index {local_rank} ({torch.cuda.get_device_name(local_rank)})")

    pg, r, world, lr = init_process_group()
    assert r == rank and lr == local_rank and world == world_size, \
        f"Mismatch env vs. init_process_group: env=({rank},{local_rank},{world_size}) pg=({r},{lr},{world})"

    # per-rank distinct buffer 
    n = 4 * 1024 * 1024  # 4M elems (~16 MB)
    x = torch.full((n,), float(rank + 1), device='cuda', dtype=torch.float32)
    print(x)
    pre_sum = x.sum().item()

    
    pg.allreduce_(x)  
    torch.cuda.synchronize()

    post_sum = x.sum().item()
    first5   = x[:5].tolist()

    print(f"[rank {rank}] | world={world} | Local grad SUM={pre_sum:.2f} | Global grad AVG(after broadcast)={post_sum:.2f} | first5={first5}")

if __name__ == "__main__":
    main()
