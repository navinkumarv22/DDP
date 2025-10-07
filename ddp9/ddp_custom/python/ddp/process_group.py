# python/ddp/process_group.py
import os, time, torch, tempfile, errno
import ddp_backend

class ProcessGroup:
    def __init__(self, rank: int, world: int, device: int, unique_id_bytes: bytes):
        self.rank = rank; self.world = world; self.device = device
        torch.cuda.set_device(device)
        opts = ddp_backend.PGOptions()
        opts.device = device; opts.rank = rank; opts.world = world
        self._pg = ddp_backend.ProcessGroupNCCL(ddp_backend.nccl_id_from_bytes(unique_id_bytes), opts)

    @property
    def world_size(self): return self._pg.world_size

    def allreduce_(self, t: torch.Tensor):
        assert t.is_cuda and t.dtype == torch.float32 and t.is_contiguous()
        self._pg.allreduce_addr_f32(int(t.data_ptr()), t.numel())

    def broadcast_(self, t: torch.Tensor, root=0):
        assert t.is_cuda and t.dtype == torch.float32 and t.is_contiguous()
        self._pg.broadcast_addr_f32(int(t.data_ptr()), t.numel(), int(root))

    def barrier(self):
        self._pg.barrier()


def _atomic_write(path: str, data: bytes):
    # Write to a temp file and rename (atomic on POSIX)
    dname = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".uid.", dir=dname)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try: os.unlink(tmp)
        except OSError: pass


def _read_exact(path: str, n: int, timeout_s: float = 30.0, poll_ms: int = 50) -> bytes:
    """Wait for file to appear and have at least n bytes; then read exactly n."""
    deadline = time.time() + timeout_s
    while True:
        try:
            st = os.stat(path)
            if st.st_size >= n:
                with open(path, "rb") as f:
                    data = f.read(n)
                    if len(data) == n:
                        return data
        except FileNotFoundError:
            pass
        if time.time() > deadline:
            raise TimeoutError(f"Timeout waiting for {path} to have {n} bytes")
        time.sleep(poll_ms / 1000.0)


def init_process_group():
    # torchrun env
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", str(rank)))

    # Rendezvous file path (deterministic per job). You can override with NCCL_BOOT_FILE.
    default_tag = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT", "29500")
    rendezvous_file = os.environ.get("NCCL_BOOT_FILE", f"/tmp/nccl_uid_{os.getuid()}_{default_tag}.bin")

    # Rank 0 writes uid once; others read it.
    if rank == 0:
        uid_bytes = ddp_backend.nccl_get_unique_id()  # 128 bytes
        _atomic_write(rendezvous_file, uid_bytes)
    # Everyone reads (rank0 will just read what it wrote; NFS delay-safe).
    uid_bytes = _read_exact(rendezvous_file, 128, timeout_s=60.0)

    # Construct NCCL PG
    pg = ProcessGroup(rank=rank, world=world, device=local, unique_id_bytes=uid_bytes)
    return pg, rank, world, local
