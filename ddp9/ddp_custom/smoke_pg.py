import os, torch
from ddp.process_group import init_process_group

pg, rank, world, local = init_process_group()
torch.cuda.set_device(local)

x = torch.zeros(4, device=f"cuda:{local}", dtype=torch.float32).contiguous()
if rank == 0:
    x.fill_(1.0)
pg.broadcast_(x, root=0)
pg.barrier()
# Now sum rank value (rank+1) into the tensor to ensure real traffic
x.add_(float(rank+1))
pg.allreduce_(x)
pg.barrier()
x /= world
print(f"rank{rank} ->", x.tolist())

