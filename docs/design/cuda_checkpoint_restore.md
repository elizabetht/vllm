# RFC: CUDA Checkpoint/Restore for Near-Zero Cold Starts

> **Issue:** [vllm-project/vllm#33930](https://github.com/vllm-project/vllm/issues/33930)
> **Scope:** v1 engine only
> **Status:** Draft

## Summary

Integrate NVIDIA's CUDA checkpoint/restore API (`cuCheckpointProcess*`) into vLLM to enable near-zero cold start times by snapshotting and restoring the full GPU state — including model weights, CUDA kernels, compiled code, CUDA graphs, streams, and contexts. This builds on vLLM's existing sleep/wake-up infrastructure.

The feature would be implemented natively in vLLM using the same dispatch chain as the existing sleep/wake-up system (`API → EngineClient → EngineCore → Executor → GPUWorker`), with no dependency on Kubernetes or external orchestrators. The CRIU + CUDA plugin integration approach is informed by [AI Dynamo's chrek](https://github.com/ai-dynamo/dynamo/tree/main/deploy/chrek), an open-source Go-based checkpoint/restore tool that has validated this pattern in production Kubernetes environments.

## Motivation

Cold start latency is a major barrier for several important use cases:

- **Multi-model serving:** Swapping between models on shared GPUs requires full teardown and re-initialization, taking 30–120s depending on model size.
- **Scale-to-zero serverless:** Platforms that scale GPU workloads to zero pay the full cold start penalty on every scale-up, making sub-minute response SLAs impossible for large models.
- **Cost-efficient hosting:** Providers cannot amortize GPU costs across many models if each model swap requires a lengthy restart, making per-request billing impractical.

Modal has demonstrated [10x cold start improvements](https://modal.com/blog/gpu-mem-snapshots) using the CUDA checkpoint API with vLLM, and InferX has reportedly built similar functionality on top of vLLM. While [AI Dynamo's chrek](https://github.com/ai-dynamo/dynamo/tree/main/deploy/chrek) provides an open-source CRIU + CUDA checkpoint orchestrator, no inference engine currently offers this as a native, built-in capability.

### What cold start currently looks like

A typical vLLM cold start involves:

1. Process startup and Python/library imports (~2-5s)
2. Model weight download/loading from disk (~5-30s depending on model size and storage)
3. Weight transfer to GPU (~2-10s)
4. `torch.compile` compilation or cache warm-up (~5-30s)
5. CUDA graph capture (~2-10s)
6. NCCL initialization for tensor parallelism (~1-3s)
7. KV cache allocation and profiling (~1-2s)

Steps 3–5 are the most expensive and are precisely what CUDA checkpoint/restore eliminates entirely.

### Relationship to sleep mode bug (#32714)

vLLM's existing sleep mode (`--enable-sleep-mode`) is broken since v0.14.0 ([#32714](https://github.com/vllm-project/vllm/issues/32714)). This RFC can be developed in parallel with that fix — they share the `SleepMode` dispatch configuration but the underlying mechanisms (`CuMemAllocator` vs `CudaCheckpointer`) are independent.

## Background

### NVIDIA CUDA Checkpoint API

NVIDIA introduced process-level CUDA checkpoint/restore in [CUDA driver 570+](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html). The API provides five operations:

| Operation | Function | Description |
|-----------|----------|-------------|
| Lock | `cuCheckpointProcessLock(pid)` | Block new CUDA API calls, wait for in-flight GPU work to complete |
| Checkpoint | `cuCheckpointProcessCheckpoint(pid)` | Copy GPU memory to host, release all GPU resources |
| Restore | `cuCheckpointProcessRestore(pid)` | Re-acquire GPUs, copy memory back, restore CUDA objects |
| Unlock | `cuCheckpointProcessUnlock(pid)` | Allow CUDA API calls to resume |
| Get State | `cuCheckpointProcessGetState(pid)` | Query current process state (RUNNING/LOCKED/CHECKPOINTED) |

The state machine is: `RUNNING → (lock) → LOCKED → (checkpoint) → CHECKPOINTED → (restore) → LOCKED → (unlock) → RUNNING`.

The [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) CLI utility wraps these APIs and can be combined with [CRIU](https://criu.org/) for full process checkpoint/restore to disk.

State preserved by CUDA checkpoint includes:

- Device memory contents (model weights, KV cache, activations)
- CUDA kernels (including JIT-compiled and `torch.compile` artifacts)
- CUDA objects (streams, contexts, events)
- Memory mappings and virtual addresses
- CUDA graphs

### CRIU (Checkpoint/Restore in Userspace)

[CRIU](https://criu.org/) is a Linux tool that freezes a running process and saves its entire CPU/OS state to disk — memory contents, file descriptors, threads, signal handlers, sockets — and can later restore the process from those image files. CRIU handles the **CPU side** of process snapshotting; the NVIDIA CUDA plugin extends CRIU to additionally checkpoint/restore GPU state during the dump/restore cycle. Together, they enable full process persistence to disk (Tier 2 of this RFC).

### vLLM's Existing Sleep/Wake-up Mechanism

vLLM already has a sleep/wake-up system in [`vllm/device_allocator/cumem.py`](../../vllm/device_allocator/cumem.py) that uses `cuMemCreate`/`cuMemMap`/`cuMemUnmap`/`cuMemRelease` to manually copy tracked allocations between GPU and CPU. The existing dispatch chain is:

```
POST /sleep
  → api_router.sleep()
    → engine_client.sleep(level)
      → engine_core_client.call_utility("sleep", level)   # ZMQ IPC
        → EngineCore.sleep(level)
          → Executor.sleep(level)
            → Executor.collective_rpc("sleep", kwargs={level: level})
              → GPUWorker.sleep(level)
                → CuMemAllocator.sleep(offload_tags=("weights",))
```

The new `suspend()`/`resume()` path will follow this identical dispatch chain.

#### Comparison: Sleep/Wake-up vs. CUDA Checkpoint

| Capability | Sleep/Wake-up | CUDA Checkpoint |
|---|---|---|
| Model weights offload to CPU | ✅ Manual per-allocation | ✅ Driver-managed |
| CUDA kernels preserved | ❌ Must reload | ✅ |
| `torch.compile` artifacts preserved | ❌ Must recompile | ✅ |
| CUDA graphs preserved | ❌ Must re-capture | ✅ |
| CUDA streams/contexts preserved | ❌ Must re-create | ✅ |
| NCCL communicators | ✅ Untouched (not in allocator pool) | ❌ Destroyed — must reinit |
| Survives process restart (via CRIU) | ❌ In-process only | ✅ |
| Can persist to disk | ❌ | ✅ |
| GPU remapping on restore | ❌ | ✅ (driver 580+) |
| Tagged/selective offload | ✅ | ❌ All-or-nothing |
| Platform requirements | Any CUDA | Linux x86_64, driver 570+ |

**Key difference:** Sleep/wake-up only manages allocations tracked by `CuMemAllocator` (weights, KV cache). NCCL communicator buffers are allocated outside this pool by PyTorch/NCCL's default allocator, so they survive sleep. CUDA checkpoint operates on the **entire process** — all GPU resources including NCCL buffers are released, requiring NCCL re-initialization on resume.

## Design

The implementation is proposed in two tiers, where Tier 1 provides immediate value and Tier 2 enables advanced scenarios.

### Tier 1: In-Process CUDA Suspend/Resume

Use `cuCheckpointProcess*` APIs to freeze and thaw GPU state while the vLLM process stays alive. This complements the existing sleep/wake-up system. The `/suspend` and `/resume` endpoints are kept separate from `/sleep` and `/wake_up` because they have different semantics — suspend destroys NCCL communicators while sleep does not, and callers need to understand this distinction.

#### Architecture

```
POST /suspend                                   POST /resume
     │                                               │
     ▼                                               ▼
api_router.suspend()                       api_router.resume()
     │                                               │
     ▼                                               ▼
engine_client.suspend()                    engine_client.resume()
     │                                               │
     ▼ (ZMQ IPC utility RPC)                         ▼
EngineCore.suspend()                       EngineCore.resume()
     │                                               │
     ▼                                               ▼
Executor.suspend()                         Executor.resume()
     │                                               │
     ▼ collective_rpc("suspend")                     ▼ collective_rpc("resume")
     │                                               │
┌────┼────────────────┐                    ┌─────────┼────────────────┐
│    ▼                │                    │         ▼                │
│ GPUWorker(rank 0)   │   ...rank N        │ GPUWorker(rank 0)  ...N │
│    │                │                    │         │                │
│    ▼                │                    │         ▼                │
│ CudaCheckpointer   │                    │ CudaCheckpointer        │
│   .suspend()        │                    │   .resume()             │
│   ├─ lock(pid)      │                    │   ├─ restore(pid)       │
│   └─ checkpoint(pid)│                    │   ├─ unlock(pid)        │
│                     │                    │   └─ reinit_nccl()      │
└─────────────────────┘                    └─────────────────────────┘
```

#### New Components

| Component | Location | Description |
|-----------|----------|-------------|
| C extension | `csrc/cuda_checkpoint.cpp` | `dlsym`-based bindings for `cuCheckpointProcess*` from `libcuda.so`, following the same pattern as `csrc/cumem_allocator.cpp`. Exposes `lock()`, `checkpoint()`, `restore()`, `unlock()`, `get_state()`, and `is_supported()` (runtime feature gate that probes driver ≥ 570). |
| Python wrapper | `vllm/device_allocator/cuda_checkpoint.py` | `CudaCheckpointer` class with `suspend(timeout_ms)` (lock + checkpoint), `resume()` (restore + unlock), `is_supported()`, and `get_state()`. Tracks `CUProcessState` {RUNNING, LOCKED, CHECKPOINTED}. All-or-nothing — no tag-based selective offload. |
| GPU Worker | `vllm/v1/worker/gpu_worker.py` | New `suspend()`, `resume()`, and `_reinit_nccl()` methods parallel to existing `sleep()`/`wake_up()`. `_reinit_nccl()` follows the elastic EP `reinitialize_distributed()` pattern: `cleanup_dist_env_and_memory()` → `init_worker_distributed_environment()` → `ensure_model_parallel_initialized()`. |
| Executor | `vllm/v1/executor/abstract.py` | `suspend()`/`resume()` via `collective_rpc` broadcast to all workers. Guards against conflicting states (sleeping vs. suspended). |
| Engine Core | `vllm/v1/engine/core.py` | `suspend()`/`resume()` delegation, dispatched as utility RPCs via `_handle_client_request(UTILITY, ...)`. |
| API endpoints | `vllm/entrypoints/serve/sleep/api_router.py` | `POST /suspend` (with `timeout_ms` query param), `POST /resume`, `GET /is_suspended`. |
| Configuration | `vllm/config.py` | New `SleepMode` enum: `disabled`, `cumem` (current behavior), `cuda_checkpoint` (new), `auto` (probe and pick best). CLI: `--sleep-mode auto`. `--enable-sleep-mode` retained as deprecated alias → `cumem`. `AUTO` resolves at startup via `CudaCheckpointer.is_supported()`. |

### Tier 2: Full Process Checkpoint to Disk (CRIU Integration)

Combine CUDA checkpoint with [CRIU](https://criu.org/) to persist the entire process state (CPU + GPU) to disk, enabling cold-start-free restarts across process boundaries. This is implemented as a Python subprocess wrapper that orchestrates `criu` and `cuda-checkpoint` CLI tools — no Go sidecar or Kubernetes controller is required. (External orchestrators like [dynamo/chrek](https://github.com/ai-dynamo/dynamo/tree/main/deploy/chrek) can build Kubernetes-native workflows on top of these primitives.)

#### Workflow

A `CheckpointOrchestrator` Python module (`vllm/checkpoint/orchestrator.py`) wraps `criu` CLI invocations:

- **`checkpoint_to_disk(output_dir)`:** Suspends via Tier 1 → performs pre-dump cleanup (remove POSIX semaphores in `/dev/shm/sem.*`, close unnecessary FDs) → writes CRIU config file with CUDA plugin settings (`enable-external-masters`, `allow-uprobes`, `tcp-close`, `skip-in-flight`) → invokes `criu dump` → saves vLLM metadata (model config, TP/PP topology, port, driver version). Optional `--leave-running` for hot snapshots.
- **`restore_from_disk(snapshot_dir, port)`:** Writes CRIU config → invokes `criu restore` → waits for readiness → signals `POST /resume` → API server re-binds to port.

Pre-dump cleanup details (learned from dynamo/chrek's production implementation):
- Remove `/dev/shm/sem.*` — CRIU cannot checkpoint named POSIX semaphores
- Set `--ghost-limit 128M` for models > 7B to avoid CRIU splitting large memory regions
- Close unnecessary file descriptors to minimize checkpoint image size

#### CLI Interface

```bash
vllm checkpoint --output /snapshots/llama-70b/               # snapshot to disk
vllm restore --snapshot /snapshots/llama-70b/ --port 8000    # restore from snapshot
vllm checkpoint --output /snapshots/llama-70b/ --leave-running  # hot snapshot
```

#### Key Challenges for Tier 2

| Challenge | Mitigation |
|-----------|------------|
| NCCL communicators destroyed | Re-initialize after restore via `_reinit_nccl()` (~1-3s). Store TP/PP topology in snapshot metadata for rebuild. |
| API server sockets dropped | CRIU config uses `tcp-close` — sockets are intentionally dropped. API server re-binds to the specified port after restore. |
| POSIX semaphores in `/dev/shm/` | Remove `sem.*` files before `criu dump` — they cannot be checkpointed. |
| File descriptors (model files, logs) | CRIU handles most FDs. Close unnecessary ones pre-dump to minimize image size. |
| Address space layout | CRIU restores at original addresses; CUDA checkpoint preserves GPU virtual addresses. |
| Different GPU on restore | Driver 580+ supports GPU UUID remapping via `CUcheckpointRestoreArgs`. |
| `torch.compile` cache paths | Must be consistent between checkpoint and restore environments — save paths in snapshot metadata. |
| Large checkpoint images | Use `--ghost-limit 128M` for models > 7B. Host must have disk space ≥ 2× GPU memory + process RSS. |

### Tier 3 (Future): Auto-Suspend and Auto-Resume

Enable vLLM to automatically suspend GPU state when idle and transparently resume when new requests arrive, eliminating the need for external orchestrators to call `/suspend` and `/resume` manually.

**How it works:** The API router runs a background idle timer. After `idle_timeout_s` (default: 300s) with no requests, the engine is suspended via Tier 1. When a new request arrives, `resume()` is called transparently before forwarding — the client experiences added latency (~3-8s) but receives a normal response. A `min_uptime_s` guard (default: 60s) prevents thrashing under bursty traffic. Requests arriving during resume are queued, not rejected.

CLI: `vllm serve --sleep-mode auto --auto-suspend --auto-suspend-timeout 300`

**Design considerations:**
- **Latency visibility:** Response header (`X-vLLM-Resumed: true`) or Prometheus metrics (`vllm:auto_resume_total`, `vllm:auto_resume_latency_seconds`)
- **Health checks while suspended:** `/health` returns `{"status": "suspended"}` so load balancers can distinguish "healthy but idle" from "healthy and serving"
- **Backend agnostic:** Auto-suspend uses whichever mode `SleepMode` resolves to (`cuda_checkpoint` or `cumem`)

**Why future:** Requires careful design around request queuing, timeout behavior, and load balancer contracts. Benefits from real-world feedback on Tier 1 manual endpoints first.

## Interaction with Existing Systems

### Sleep/Wake-up Coexistence

The existing sleep/wake-up system and CUDA checkpoint serve complementary roles:

- **sleep/wake-up (`cumem` mode)**: Faster to enter sleep (~1-2s). Fine-grained control via tags (offload weights only, keep KV cache, etc.). Works on all CUDA platforms. NCCL communicators survive. But: CUDA graphs, compiled kernels, `torch.compile` artifacts are all lost and must be re-created on wake-up.
- **CUDA checkpoint (`cuda_checkpoint` mode)**: Slower to checkpoint (~2-5s) but much faster to resume (no `torch.compile` or CUDA graph re-warmup). All-or-nothing. NCCL must be re-initialized (~1-3s). Linux x86_64 + driver 570+ only.

Both coexist under the `SleepMode` enum. The `AUTO` mode selects `cuda_checkpoint` if the platform supports it, otherwise falls back to `cumem`.

| Operation | `cumem` mode | `cuda_checkpoint` mode |
|-----------|-------------|----------------------|
| Enter sleep/suspend | ~1-2s (cudaMemcpy per-allocation) | ~2-5s (driver-managed, all GPU resources) |
| Resume/wake-up | ~5-30s (cudaMemcpy + torch.compile + CUDA graphs) | ~2-5s (driver restore + NCCL reinit) |
| **Total round-trip** | **~6-32s** | **~4-10s** |

### CUDA Graphs

With sleep/wake-up, CUDA graphs must be re-captured after wake-up, which is expensive (~2-10s). With CUDA checkpoint, CUDA graphs are preserved across suspend/resume — this is one of the largest performance wins.

### `torch.compile`

Currently, vLLM caches `torch.compile` artifacts on disk (`~/.cache/vllm/torch_compile_cache/`), but compilation still takes significant time on cold start (~5-30s). With CUDA checkpoint, the already-compiled GPU kernels are preserved in the checkpoint, eliminating recompilation entirely.

### Model Swapping

CUDA checkpoint enables a model swapping pattern: suspend model A (~2-5s, GPU → host), then either initialize a new engine with model B or restore a previously checkpointed model B. Combined with Tier 2 disk persistence, this allows pre-warmed model snapshots to be swapped in without any cold start penalty.

### NCCL Communicator Lifecycle

This is the most significant behavioral difference from the existing sleep/wake-up system:

| | `sleep()`/`wake_up()` | `suspend()`/`resume()` |
|---|---|---|
| NCCL state | **Preserved** — NCCL buffers are outside `CuMemAllocator`'s pool | **Destroyed** — `cuCheckpointProcessCheckpoint` releases ALL GPU resources |
| Reinit needed? | No | Yes — `_reinit_nccl()` after every `resume()` |
| Reinit cost | N/A | ~1-3s (TP/PP group rebuild via TCP rendezvous) |
| Pattern reference | N/A | `reinitialize_distributed()` / `cleanup_dist_env_and_memory()` in elastic EP |

#### Reinit Sequence

NNCL process groups (`_TP`, `_PP`, `_DP`, `_EP`, etc.) are module-level globals in `parallel_state.py`, each owning a NCCL `ProcessGroup` (`ncclComm_t` + GPU buffers), a Gloo `ProcessGroup` (CPU-side), and a `CudaCommunicator` (`PyNcclCommunicator` + `CustomAllreduce` IPC buffers). All GPU-side resources are destroyed by `cuCheckpointProcessCheckpoint`.

The reinit follows the elastic EP `reinitialize_distributed()` pattern:

1. **Before suspend:** `torch.cuda.synchronize()` → `cleanup_dist_env_and_memory()` (destroys all groups, calls `gc.collect()`) → `CudaCheckpointer.suspend()`
2. **After resume:** `CudaCheckpointer.resume()` → `torch.cuda.set_device(local_rank)` → `init_worker_distributed_environment()` (TCP rendezvous + `ncclCommInitRank` for each group) → `ensure_model_parallel_initialized(tp, pp)`

All steps are broadcast via `collective_rpc` — `destroy_process_group()` is a collective operation requiring all ranks to participate.

**Estimated cost:** ~1-3s total for TP=4/8 (dominated by `ncclCommInitRank` GPU buffer allocation + topology discovery).

**Key pitfall — CUDA graphs with NCCL:** CUDA graphs containing NCCL collectives embed references to old `ncclComm_t` handles. After NCCL rebuild, these are stale — such graphs likely need re-capture even with CUDA checkpoint. This is an open investigation item for Phase 1.

## Platform Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| OS | Linux | Linux |
| Architecture | x86_64 | x86_64 |
| NVIDIA Driver | 570 | 580+ (GPU remapping support) |
| CUDA Toolkit | 12.8+ | 13.0+ |
| Host RAM | ≥ GPU memory used by model | ≥ 2× GPU memory (headroom for checkpoint + OS) |
| CRIU (Tier 2 only) | 4.0 | 4.2+ (with CUDA plugin) |
| Disk (Tier 2 only) | ≥ 2× GPU memory + process RSS | Fast NVMe SSD recommended |
| GPU Memory | UVM/IPC not supported | Standard allocations only |

> **Host RAM Warning:** CUDA checkpoint copies all GPU memory to host RAM during suspend. For a 70B model (~140 GiB in FP16), the host must have at least 140 GiB of free RAM. The `suspend()` implementation should validate available host memory before proceeding and raise a clear error if insufficient.

## Implementation Plan

### Phase 1: Core Infrastructure (Tier 1)

1. **C extension** (`csrc/cuda_checkpoint.cpp`) — `dlsym`-based bindings for `cuCheckpointProcess*`, plus `is_supported()` runtime probe
2. **Python wrapper** (`vllm/device_allocator/cuda_checkpoint.py`) — `CudaCheckpointer` class with `suspend()`/`resume()`/`is_supported()`
3. **`SleepMode` enum** (`vllm/config.py`) — replace boolean `enable_sleep_mode`, add `--sleep-mode` CLI arg with deprecated alias
4. **GPU Worker** (`vllm/v1/worker/gpu_worker.py`) — `suspend()`/`resume()`/`_reinit_nccl()` methods
5. **Unit tests** with a small model (e.g., `facebook/opt-125m`) on single GPU

### Phase 2: Engine Integration (Tier 1)

1. **Executor** (`vllm/v1/executor/abstract.py`) — `suspend()`/`resume()` with `collective_rpc` broadcast and `is_suspended` state
2. **Engine Core** (`vllm/v1/engine/core.py`) — `suspend()`/`resume()` delegation as utility RPCs
3. **Engine Client** protocol — `suspend()`/`resume()`/`is_suspended()` async wrappers
4. **API endpoints** (`vllm/entrypoints/serve/sleep/api_router.py`) — `POST /suspend`, `POST /resume`, `GET /is_suspended`
5. **Request draining** — ensure all in-flight requests complete before suspend
6. **Integration tests** — streaming, tool calls, multi-GPU (TP=2, TP=4)

### Phase 3: Optimization and Polish

1. **NCCL reinit benchmarking** — measure reinit cost at TP=2, 4, 8 to validate it doesn't negate checkpoint benefits
2. **`AUTO` mode** — runtime capability probe, automatic selection
3. **Host RAM validation** — pre-suspend check with clear error messages
4. **CLI support** — `vllm serve --sleep-mode auto`
5. **Benchmarking** — cold start times for various model sizes (0.5B, 7B, 13B, 70B) with and without checkpoint
6. **Documentation** and deployment guides

### Phase 4: CRIU Integration (Tier 2)

1. **`CheckpointOrchestrator`** (`vllm/checkpoint/orchestrator.py`) — Python subprocess wrapper for `criu dump`/`criu restore` with CUDA plugin config
2. **Pre-dump cleanup** — semaphore removal, FD cleanup, ghost-limit tuning
3. **CLI commands** — `vllm checkpoint` / `vllm restore`
4. **Snapshot metadata** format — model config, TP/PP topology, `torch.compile` cache paths, port, driver version
5. **Multi-node** checkpoint/restore for tensor-parallel across hosts (future)

### Phase 5: Auto-Suspend/Auto-Resume (Tier 3, Future)

1. **Idle timer** in API router — background task that monitors last-request timestamp and triggers `suspend()` after configurable timeout
2. **Transparent resume middleware** — intercept incoming requests, call `resume()` if engine is suspended, queue requests during resume
3. **Thrashing prevention** — minimum uptime guard, exponential backoff on rapid suspend/resume cycles
4. **Health check contract** — define `/health` behavior while suspended (return `{"status": "suspended"}` for load balancer awareness)
5. **Observability** — Prometheus metrics for auto-suspend/resume events, latency, idle time
6. **`AutoSuspendConfig`** (`vllm/config.py`) — `--auto-suspend`, `--auto-suspend-timeout`, `--auto-suspend-min-uptime` CLI flags
7. **Integration tests** — bursty traffic patterns, concurrent resume races, load balancer compatibility

## Open Questions

1. **NCCL rebuild cost:** How expensive is NCCL communicator re-initialization after restore? Does it negate the checkpoint benefit for large TP configurations (e.g., TP=8)? The `_reinit_dp_group()` pattern from elastic EP suggests ~1-3s, but this needs benchmarking at scale. If too costly, investigate marking NCCL resources as external to the checkpoint (requires driver support).

2. **Host RAM validation strategy:** Should `suspend()` check `psutil.virtual_memory().available` against total GPU allocation size and refuse to proceed if insufficient? Or should we let the driver fail and catch the error? Recommend: validate upfront with a clear error message.

3. **Interaction with `torch.compile` cache:** If the `torch.compile` disk cache (`~/.cache/vllm/torch_compile_cache/`) is already populated, does CUDA checkpoint still provide meaningful speedup over `cumem` sleep + cache-warm compile? Needs benchmarking, but CUDA checkpoint still eliminates the compilation step itself (loading from cache is fast, but not free).

4. **UVM allocations:** The CUDA checkpoint API does not support Unified Virtual Memory (UVM) or IPC memory. Are there vLLM allocations using UVM that would need to be refactored? An audit of allocation patterns is needed.

5. **Multi-model orchestration:** Should vLLM include a built-in model multiplexer that leverages checkpoint/restore to swap models, or should this be left to external orchestrators (like LiteLLM, dynamo, etc.)? Recommend: leave to external orchestrators — vLLM provides the `/suspend`/`resume` primitives, orchestrators build workflows.

6. **Backward compatibility:** The `--enable-sleep-mode` flag is widely used. The migration to `--sleep-mode cumem` must be backward-compatible with a deprecation warning.

7. **Auto-suspend request queuing:** When a request arrives during auto-resume, should it block (simple, but adds latency) or be queued with a timeout (more complex, but better for concurrent arrivals)? What should the maximum queue depth be? Should the first request that triggers resume receive an `X-vLLM-Resumed: true` header to give clients visibility into the added latency?

8. **CUDA graphs with rebuilt NCCL:** CUDA checkpoint preserves CUDA graphs at the driver level, but graphs containing NCCL collectives (allreduce, allgather) embed references to old `ncclComm_t` handles. After NCCL rebuild, these references are stale — do such graphs need re-capture even with CUDA checkpoint? This would reduce the speedup for TP>1 workloads and needs benchmarking in Phase 1.

## Prior Art

- [Modal GPU Memory Snapshots](https://modal.com/blog/gpu-mem-snapshots) — Demonstrated 10x cold start improvements using CUDA checkpoint with vLLM in production.
- [InferX](https://inferx.net/) — Reportedly built CUDA checkpoint on top of vLLM (proprietary).
- [dynamo/chrek](https://github.com/ai-dynamo/dynamo/tree/main/deploy/chrek) — Open-source Go-based CRIU + CUDA checkpoint orchestrator for Kubernetes. Used as reference for CRIU config file format, pre-dump cleanup (semaphore removal), and CUDA plugin settings. chrek targets containerized Kubernetes workloads; this RFC targets native vLLM integration.
- [Aegaeon](https://ennanzhai.github.io/pub/sosp25-aegaeon.pdf) (SOSP'25) — Academic paper on inference-engine reuse and dynamic weight/KV-cache swapping for GPU pooling.

## References

- [NVIDIA CUDA Checkpoint API Documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CHECKPOINT.html)
- [NVIDIA cuda-checkpoint CLI utility](https://github.com/NVIDIA/cuda-checkpoint)
- [Modal: GPU Memory Snapshots blog post](https://modal.com/blog/gpu-mem-snapshots)
- [CRIU — Checkpoint/Restore in Userspace](https://criu.org/Main_Page)
- [vLLM sleep/wake-up implementation](../../vllm/device_allocator/cumem.py)
- [vLLM CUDA graphs design doc](cuda_graphs.md)
- [vLLM torch.compile design doc](torch_compile.md)
- [vLLM sleep mode bug #32714](https://github.com/vllm-project/vllm/issues/32714)
