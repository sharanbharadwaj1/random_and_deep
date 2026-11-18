"""
Toy KV-cache allocator + continuous batching scheduler (vLLM-style).

Models MEMORY ONLY: no attention, no real compute. It only counts how many
prompt tokens would need prefill vs. be reused from the prefix cache.

Prefix cache = flat dict keyed by CHAINED block hashes (vLLM-style), standing
in for a radix tree. Only full blocks are shared; tails and decode blocks
are private to a request.
"""
import argparse
from typing import Dict, List, Optional, Tuple

BLOCK_SIZE = 16
LLAMA_3_8B = {"layers": 32, "kv_heads": 8, "head_dim": 128}
FP16_BYTES = 2

Entry = Tuple[Optional[int], int]  # (cache key or None if private, physical block id)


def bytes_per_token(cfg, precision_bytes=FP16_BYTES) -> int:
    # key + value, per kv head, per layer
    return 2 * cfg["kv_heads"] * cfg["head_dim"] * precision_bytes * cfg["layers"]


class PagedMemoryManager:
    """Fixed pool of physical KV blocks with a free list."""

    def __init__(self, total_vram_gb: float, block_size: int, token_bytes: int):
        self.block_size = block_size
        self.bytes_per_block = token_bytes * block_size
        self.total_blocks = int(total_vram_gb * (1024 ** 3) // self.bytes_per_block)
        self.free_blocks = list(range(self.total_blocks))

    def allocate_block(self) -> Optional[int]:
        return self.free_blocks.pop() if self.free_blocks else None

    def free_block(self, block_id: int):
        self.free_blocks.append(block_id)

    @property
    def used_blocks(self) -> int:
        return self.total_blocks - len(self.free_blocks)


class PrefixCache:
    """Flat dict standing in for a radix tree.

    key = hash((parent_key, tokens_in_block)), so a block only matches if the
    ENTIRE prefix before it matches too. Same idea as vLLM's chained hashes.
    """

    def __init__(self, manager: PagedMemoryManager, enable_sharing: bool = True):
        self.m = manager
        self.enable_sharing = enable_sharing
        self.cache: Dict[int, dict] = {}
        self.prefill_computed = 0  # prompt tokens that would need real compute
        self.prefill_reused = 0    # prompt tokens served from shared blocks

    def allocate_prompt(self, tokens: List[str]) -> Optional[List[Entry]]:
        bs = self.m.block_size
        taken: List[Entry] = []
        computed = reused = 0
        prev = 0
        for i in range(0, len(tokens), bs):
            chunk = tuple(tokens[i:i + bs])
            if self.enable_sharing and len(chunk) == bs:  # only FULL blocks are cacheable
                key = hash((prev, chunk))
                prev = key
                hit = self.cache.get(key)
                if hit:
                    hit["ref"] += 1
                    taken.append((key, hit["block"]))
                    reused += bs
                    continue
                bid = self.m.allocate_block()
                if bid is None:
                    self.release(taken)  # roll back a half-finished admission
                    return None
                self.cache[key] = {"block": bid, "ref": 1}
                taken.append((key, bid))
            else:  # partial tail (or sharing disabled): private block
                bid = self.m.allocate_block()
                if bid is None:
                    self.release(taken)
                    return None
                taken.append((None, bid))
            computed += len(chunk)
        self.prefill_computed += computed
        self.prefill_reused += reused
        return taken

    def release(self, entries: List[Entry]):
        """Exact inverse of allocation: drop precisely the blocks we were given."""
        for key, bid in entries:
            if key is None:
                self.m.free_block(bid)
            else:
                e = self.cache[key]
                e["ref"] -= 1
                if e["ref"] == 0:
                    self.m.free_block(e["block"])
                    del self.cache[key]


class Request:
    def __init__(self, rid: str, prompt: List[str], target_len: int):
        self.rid = rid
        self.prompt = prompt
        self.tokens = prompt.copy()
        self.target_len = target_len
        self.generated = 0
        self.blocks: List[Entry] = []
        self.status = "Waiting"


class Scheduler:
    def __init__(self, manager: PagedMemoryManager, enable_sharing: bool):
        self.m = manager
        self.cache = PrefixCache(manager, enable_sharing)
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.tick = 0
        self.preemptions = 0

    def add(self, req: Request):
        self.waiting.append(req)

    def _preempt(self, req: Request):
        """Recompute-style preemption: free everything, requeue at the front."""
        self.cache.release(req.blocks)
        req.blocks, req.tokens, req.generated = [], req.prompt.copy(), 0
        req.status = "Waiting"
        self.running.remove(req)
        self.waiting.insert(0, req)
        self.preemptions += 1

    def step(self):
        self.tick += 1
        # 1. Admit in FIFO order; stop at the first request that doesn't fit.
        for req in list(self.waiting):
            blocks = self.cache.allocate_prompt(req.tokens)
            if blocks is None:
                break
            req.blocks, req.status = blocks, "Running"
            self.waiting.remove(req)
            self.running.append(req)

        # 2. Decode one token per running request.
        finished = []
        for req in list(self.running):
            if req.status != "Running":  # preempted earlier this tick
                continue
            if len(req.tokens) >= len(req.blocks) * self.m.block_size:  # need a new block
                bid = self.m.allocate_block()
                while bid is None:
                    victim = self.running[-1]  # preempt the youngest
                    self._preempt(victim)
                    if victim is req:
                        break
                    bid = self.m.allocate_block()
                if req.status != "Running":
                    continue
                req.blocks.append((None, bid))  # decode blocks are private
            req.tokens.append(f"gen_{req.generated}")
            req.generated += 1
            if req.generated >= req.target_len:
                finished.append(req)

        # 3. Free finished requests with the exact blocks they hold.
        for req in finished:
            req.status = "Finished"
            self.cache.release(req.blocks)
            req.blocks = []
            self.running.remove(req)

    def wasted_slots(self) -> int:
        bs = self.m.block_size
        return sum((-len(r.tokens)) % bs for r in self.running)


def run(sharing: bool, a) -> dict:
    tb = bytes_per_token(LLAMA_3_8B)
    mgr = PagedMemoryManager(a.vram_gb, BLOCK_SIZE, tb)
    sched = Scheduler(mgr, sharing)
    system = ["system_token"] * a.prefix
    for i in range(a.users):
        sched.add(Request(f"User_{i}", system + [f"user_{i}_q"] * a.query, a.output))

    peak_active = peak_used = 0
    while sched.running or sched.waiting:
        sched.step()
        peak_active = max(peak_active, len(sched.running))
        peak_used = max(peak_used, mgr.used_blocks)
        if a.verbose:
            used = mgr.used_blocks
            frag = 100 * sched.wasted_slots() / (used * BLOCK_SIZE) if used else 0
            print(f"Tick {sched.tick:03d} | Active {len(sched.running)} | "
                  f"Queue {len(sched.waiting)} | Blocks {used}/{mgr.total_blocks} | Frag {frag:.1f}%")
        assert sched.tick < 100000, "scheduler stuck"

    # Leak check: with symmetric alloc/free these must both be zero.
    assert mgr.used_blocks == 0 and not sched.cache.cache, "LEAK"
    return {
        "pool_blocks": mgr.total_blocks,
        "ticks": sched.tick,
        "peak_active": peak_active,
        "peak_blocks": peak_used,
        "prefill_computed": sched.cache.prefill_computed,
        "prefill_reused": sched.cache.prefill_reused,
        "preemptions": sched.preemptions,
        "end_blocks": mgr.used_blocks,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=5)
    p.add_argument("--prefix", type=int, default=1000)
    p.add_argument("--query", type=int, default=20)
    p.add_argument("--output", type=int, default=30)
    p.add_argument("--vram-gb", type=float, default=0.20)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    print(f"KV bytes/token: {bytes_per_token(LLAMA_3_8B)} | users={a.users} "
          f"prefix={a.prefix} query={a.query} output={a.output} pool={a.vram_gb}GB\n")
    for label, sharing in (("NO sharing", False), ("WITH prefix sharing", True)):
        r = run(sharing, a)
        print(f"== {label} ==")
        for k, v in r.items():
            print(f"  {k}: {v}")
        print()


#to run in Notebook use the below commands
# !python kv_sim.py
# !python kv_sim.py --users 52 --prefix 2000 --query 50 --output 100 --vram-gb 0.5
