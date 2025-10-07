# python/ddp/ddp.py
# -----------------------------------------------------------------------------
# DDP with bucketization + CHUNKED parameters.
# Supports PG collectives in both Tensor and address forms.
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
import torch
from torch import nn

# ----------------------------- Public spec -----------------------------------

@dataclass
class ParamSpec:
    length_elems: int
    device: int
    p_ref: nn.Parameter

# --------------------------- Internal structs --------------------------------

@dataclass
class _Chunk:
    bucket_id: int
    offset: int
    length: int

@dataclass  
class _Bucket:
    buffer: torch.Tensor   # 1D fp32 CUDA tensor
    total_chunks: int
    ready_chunks: int = 0
    reduced: bool = False

# ------------------------------ Helpers --------------------------------------

def _maybe_call(x):
    return x() if callable(x) else x

def _flat_fp32(x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.float32: x = x.float()
    if not x.is_contiguous(): x = x.contiguous()
    return x.view(-1)

def _pick(pg, names: List[str]) -> Optional[Callable]:
    for n in names:
        fn = getattr(pg, n, None)
        if fn is not None:
            return fn
    return None

# ------------------------------- DDP core ------------------------------------

class DistributedDataParallel:
    def __init__(self, param_specs: List[ParamSpec], process_group, bucket_bytes: int = 25 * 1024 * 1024):
        assert bucket_bytes > 0
        self.pg = process_group

        ws = getattr(process_group, "world_size", None) or getattr(process_group, "world", None)
        assert ws is not None, "ProcessGroup must expose world_size/world"
        self.world_size: int = int(_maybe_call(ws))
        assert self.world_size >= 1

        self.param_specs = param_specs
        self.bucket_cap_elems = max(1, bucket_bytes // 4)

        self.buckets: List[_Bucket] = []
        self.param_chunks: Dict[int, List[_Chunk]] = {}
        self._ready_queue: List[int] = []

        self._build_buckets_with_chunking()
        self._register_hooks()

        # Detect PG methods (Tensor-first, then addr forms, then underscore names)
        self._ar_tensor = _pick(self.pg, ["allreduce_", "allreduce_tensor", "allreduce"])
        self._bc_tensor = _pick(self.pg, ["broadcast_", "broadcast_tensor", "broadcast"])

        self._ar_addr = _pick(self.pg, ["allreduce_addr_f32", "allreduce_addr", "allreduce_float32"])
        self._bc_addr = _pick(self.pg, ["broadcast_addr_f32", "broadcast_addr", "broadcast_float32"])

        # rank for emulate path
        self._rank_fn = _pick(self.pg, ["rank"]) or (lambda: getattr(self.pg, "rank", 0))

        if self._ar_tensor is None and self._ar_addr is None:
            raise RuntimeError(f"ProcessGroup missing allreduce method. Available: {dir(self.pg)}")

    # ---------------------- bucketization with chunking -----------------------

    def _new_bucket(self, device_index: int) -> int:
        buf = torch.empty(self.bucket_cap_elems, device=f"cuda:{device_index}", dtype=torch.float32)
        self.buckets.append(_Bucket(buffer=buf, total_chunks=0))
        return len(self.buckets) - 1

    def _build_buckets_with_chunking(self):
        cur_bid = None
        used = 0
        for ps in reversed(self.param_specs):
            dev = int(ps.device)
            length = int(ps.length_elems)
            self.param_chunks[id(ps.p_ref)] = []

            rem = length
            while rem > 0:
                need_new = (
                    cur_bid is None
                    or self.buckets and self.buckets[cur_bid].buffer.device.index != dev
                    or used == self.bucket_cap_elems
                )
                if need_new:
                    cur_bid = self._new_bucket(dev)
                    used = 0
                take = min(self.bucket_cap_elems - used, rem)
                self.param_chunks[id(ps.p_ref)].append(_Chunk(bucket_id=cur_bid, offset=used, length=take))
                used += take
                rem  -= take
                self.buckets[cur_bid].total_chunks += 1

    # ---------------------------- autograd hooks ------------------------------

    def _register_hooks(self):
        for ps in self.param_specs:
            p = ps.p_ref
            if not isinstance(p, nn.Parameter) or not p.requires_grad:
                continue
            def make_hook(ps_local: ParamSpec):
                def _hook(grad: torch.Tensor):
                    return self._hook_copy_chunks(ps_local, grad)
                return _hook
            p.register_hook(make_hook(ps))

    def _hook_copy_chunks(self, ps: ParamSpec, grad: torch.Tensor):
        g = _flat_fp32(grad)
        if g.numel() == 0:
            return grad
        pos = 0
        for ch in self.param_chunks[id(ps.p_ref)]:
            b = self.buckets[ch.bucket_id]
            l = ch.length
            b.buffer[ch.offset: ch.offset + l].copy_(g[pos: pos + l], non_blocking=True)
            b.ready_chunks += 1
            pos += l
            if b.ready_chunks == b.total_chunks and not b.reduced:
                self._ready_queue.append(ch.bucket_id)
        return grad

    # ----------------------- collectives & step control -----------------------

    def _allreduce_sum(self, t1d_fp32: torch.Tensor):
        """Try Tensor API first; fall back to address API."""
        assert t1d_fp32.dtype == torch.float32 and t1d_fp32.is_contiguous()
        if self._ar_tensor is not None:
            try:
                # Most wrappers expose allreduce_(tensor)
                self._ar_tensor(t1d_fp32)
                return
            except TypeError:
                # Some expose (tensor, op) etc.; not supported here—keep simple
                pass
        if self._ar_addr is not None:
            addr = t1d_fp32.data_ptr()
            count = t1d_fp32.numel()
            try:
                self._ar_addr(addr, count)
                return
            except TypeError as e:
                raise RuntimeError(f"allreduce addr binding signature mismatch: {self._ar_addr} :: {e}")
        raise RuntimeError("No usable allreduce found (tensor or addr forms).")

    def synchronize(self):
        if not self._ready_queue:
            return

        # Our ProcessGroup exposes barrier() which synchronizes the NCCL stream.
        # We call it once to make sure all enqueued allreduces are done.
        if hasattr(self.pg, "barrier"):
            self.pg.barrier()

        # For each ready bucket, copy the averaged (fp32) slices back into per-param grads.
        with torch.no_grad():
            # NOTE: We scatter by visiting each param's chunks that belong to this bucket.
            # param.grad is created lazily if missing.
            ready_now = list(self._ready_queue)
            self._ready_queue.clear()

            for bid in ready_now:
                b = self.buckets[bid]
                if b.reduced:
                    continue

                # For every parameter that has slices in this bucket:
                for ps in self.param_specs:
                    p = ps.p_ref
                    chs = self.param_chunks.get(id(p), ())
                    if not chs:
                        continue

                    # Ensure grad buffer exists (match param dtype/device, but write via fp32 flat first)
                    if p.grad is None:
                        p.grad = torch.empty_like(p.data)

                    # Flatten a fp32 view for scattering; keep on same device
                    gflat_fp32 = p.grad.detach().to(dtype=torch.float32, device=p.device).view(-1)

                    for ch in chs:
                        if ch.bucket_id != bid:
                            continue
                        off, ln = ch.offset, ch.length
                        gflat_fp32[:ln].copy_(b.buffer[off: off + ln], non_blocking=True)

                        # If original grad dtype != fp32, cast back in place
                        if p.grad.dtype != torch.float32:
                            # reshape view to original and cast back
                            p.grad.copy_(gflat_fp32.view_as(p.grad).to(p.grad.dtype), non_blocking=True)
                        else:
                            # write back the updated flat slice into p.grad (same dtype)
                            # (when dtype is fp32 the gflat_fp32 is already a view equivalent)
                            p.grad.view(-1)[0:ln] = gflat_fp32[0:ln]

                b.reduced = True


    def step_reset(self):
        for b in self.buckets:
            b.ready_chunks = 0
            b.reduced = False
        self._ready_queue.clear()

    # -------------------------------- broadcast -------------------------------

    @torch.no_grad()
    def broadcast_parameters(self, root: int = 0):
        """
        Broadcast parameters from root to others.
        Supports:
          - broadcast_(tensor)
          - broadcast_(tensor, root)
          - broadcast_addr*(addr, count[, root])
        Else emulates with allreduce(sum) + zeroing on non-root.
        """
        my_rank = 0
        if self._rank_fn:
            try:
                my_rank = int(_maybe_call(self._rank_fn))
            except Exception:
                my_rank = 0

        def _try_broadcast_tensor(t: torch.Tensor) -> bool:
            if self._bc_tensor is None:
                return False
            # Try (tensor, root)
            try:
                self._bc_tensor(t, int(root))
                return True
            except TypeError:
                pass
            # Try (tensor)
            try:
                self._bc_tensor(t)
                return True
            except TypeError:
                return False

        def _try_broadcast_addr(t: torch.Tensor) -> bool:
            if self._bc_addr is None:
                return False
            addr = t.data_ptr(); count = t.numel()
            # Try (addr, count, root)
            try:
                self._bc_addr(addr, count, int(root))
                return True
            except TypeError:
                pass
            # Try (addr, count)
            try:
                self._bc_addr(addr, count)
                return True
            except TypeError:
                return False

        for ps in self.param_specs:
            p = ps.p_ref
            dev = ps.device
            flat = _flat_fp32(p.detach().to(device=f"cuda:{dev}"))

            ok = _try_broadcast_tensor(flat) or _try_broadcast_addr(flat)
            if not ok:
                # Fallback: emulate via allreduce(sum)
                if my_rank != int(root):
                    flat.zero_()
                self._allreduce_sum(flat)

            # Copy back to original dtype/device if needed
            if p.dtype != torch.float32 or p.device.index != dev:
                p.data.copy_(flat.view_as(p).to(dtype=p.dtype, device=p.device), non_blocking=True)
            else:
                p.data.copy_(flat.view_as(p), non_blocking=True)

        torch.cuda.synchronize()
