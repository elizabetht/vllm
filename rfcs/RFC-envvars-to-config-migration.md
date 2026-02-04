# RFC: Migrate Environment Variables to Typed Configuration Objects

**Status**: Draft  
**Author**: @elizabeththomas  
**Created**: 2026-02-04  
**Issue**: [Link to issue about environment variable proliferation]

## Summary

Migrate a significant portion of vLLM's environment variables to properly typed configuration objects. Environment variables have proliferated across the codebase and now control critical runtime behavior, leading to maintainability issues, lack of type safety, and poor discoverability.

## Motivation

### Current Problems

1. **Global State Pollution**: Environment variables are equivalent to global variables, making it difficult to:
   - Test different configurations in the same process
   - Understand dependencies between components
   - Track where configuration values are used

2. **No Type Safety**: Environment variables are strings with no validation, leading to:
   - Runtime errors from invalid values
   - Silent failures from typos
   - No IDE autocomplete or type checking

3. **Poor Discoverability**: Users must read documentation or source code to find available options:
   - No structured --help output
   - No validation of allowed values
   - Difficult to understand relationships between options

4. **No Hierarchy**: Flat namespace makes it hard to:
   - Organize related settings
   - Apply settings to specific components
   - Override defaults at different levels

5. **Testing Challenges**: Tests must manipulate global environment state:
   - Requires setup/teardown of env vars
   - Cannot easily run tests with different configs in parallel
   - Hard to mock or inject test configurations

### Examples of Problematic Usage

```python
# Current: Hard to test, no type safety
if envs.VLLM_KV_CACHE_LAYOUT == "NHD":
    layout = KVCacheLayout.NHD
    
# Proposed: Type-safe, testable
if config.cache_config.kv_cache_layout == KVCacheLayout.NHD:
    layout = config.cache_config.kv_cache_layout
```

## Proposal

### Phase 1: High Priority Migrations (v0.8.0)

Move configuration that controls core runtime behavior to typed config objects.

#### 1.1 Extend `CacheConfig`

```python
@dataclass
class CacheConfig:
    # Existing fields...
    
    # New fields (migrated from envvars)
    kv_cache_layout: Literal["NHD", "HND"] | None = None  # from VLLM_KV_CACHE_LAYOUT
    cpu_kv_cache_space_gb: int | None = None  # from VLLM_CPU_KVCACHE_SPACE
    allow_chunked_local_attn_with_hybrid: bool = True  # from VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE
    flashinfer_workspace_buffer_size: int = 394 * 1024 * 1024  # from VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
```

#### 1.2 Extend `ParallelConfig`

```python
@dataclass
class ParallelConfig:
    # Existing fields...
    
    # New fields (migrated from envvars)
    ray_compiled_dag_channel_type: Literal["auto", "nccl", "shm"] = "auto"  # from VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
    ray_compiled_dag_overlap_comm: bool = False  # from VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM
    ray_dp_pack_strategy: Literal["strict", "fill", "span"] = "strict"  # from VLLM_RAY_DP_PACK_STRATEGY
    worker_multiproc_method: Literal["fork", "spawn"] = "fork"  # from VLLM_WORKER_MULTIPROC_METHOD
    allreduce_use_symm_mem: bool = True  # from VLLM_ALLREDUCE_USE_SYMM_MEM
    pp_layer_partition: str | None = None  # from VLLM_PP_LAYER_PARTITION
    mq_max_chunk_bytes_mb: int = 16  # from VLLM_MQ_MAX_CHUNK_BYTES_MB
    deepep_buffer_size_mb: int = 1024  # from VLLM_DEEPEP_BUFFER_SIZE_MB
```

#### 1.3 Extend `CompilationConfig`

```python
@dataclass
class CompilationConfig:
    # Existing fields...
    
    # New fields (migrated from envvars)
    enable_inductor_max_autotune: bool = True  # from VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE
    enable_inductor_coordinate_descent_tuning: bool = True  # from VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING
    compile_cache_save_format: Literal["binary", "unpacked"] = "binary"  # from VLLM_COMPILE_CACHE_SAVE_FORMAT
```

#### 1.4 Extend `KVTransferConfig`

```python
@dataclass
class KVTransferConfig:
    # Existing fields...
    
    # Connector-specific configurations
    nixl_side_channel_host: str = "localhost"  # from VLLM_NIXL_SIDE_CHANNEL_HOST
    nixl_side_channel_port: int = 5600  # from VLLM_NIXL_SIDE_CHANNEL_PORT
    nixl_abort_request_timeout: int = 480  # from VLLM_NIXL_ABORT_REQUEST_TIMEOUT
    
    moriio_connector_read_mode: bool = False  # from VLLM_MORIIO_CONNECTOR_READ_MODE
    moriio_qp_per_transfer: int = 1  # from VLLM_MORIIO_QP_PER_TRANSFER
    moriio_post_batch_size: int = -1  # from VLLM_MORIIO_POST_BATCH_SIZE
    moriio_num_workers: int = 1  # from VLLM_MORIIO_NUM_WORKERS
    
    mooncake_abort_request_timeout: int = 480  # from VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
    mooncake_bootstrap_port: int = 8998  # from VLLM_MOONCAKE_BOOTSTRAP_PORT
```

#### 1.5 Create `MultiModalConfig`

```python
@dataclass
class MultiModalConfig:
    """Configuration for multimodal inputs (images, video, audio)."""
    
    media_connector: str = "http"  # from VLLM_MEDIA_CONNECTOR
    video_loader_backend: str = "opencv"  # from VLLM_VIDEO_LOADER_BACKEND
    
    image_fetch_timeout: int = 5  # from VLLM_IMAGE_FETCH_TIMEOUT
    video_fetch_timeout: int = 30  # from VLLM_VIDEO_FETCH_TIMEOUT
    audio_fetch_timeout: int = 10  # from VLLM_AUDIO_FETCH_TIMEOUT
    
    media_url_allow_redirects: bool = True  # from VLLM_MEDIA_URL_ALLOW_REDIRECTS
    media_loading_thread_count: int = 8  # from VLLM_MEDIA_LOADING_THREAD_COUNT
    max_audio_clip_filesize_mb: int = 25  # from VLLM_MAX_AUDIO_CLIP_FILESIZE_MB
    mm_hasher_algorithm: str = "blake3"  # from VLLM_MM_HASHER_ALGORITHM
```

#### 1.6 Extend `ModelConfig`

```python
@dataclass
class ModelConfig:
    # Existing fields...
    
    # New fields (migrated from envvars)
    float32_matmul_precision: Literal["highest", "high", "medium"] = "highest"  # from VLLM_FLOAT32_MATMUL_PRECISION
    fused_moe_chunk_size: int = 16 * 1024  # from VLLM_FUSED_MOE_CHUNK_SIZE
    enable_fused_moe_activation_chunking: bool = True  # from VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING
    moe_dp_chunk_size: int = 256  # from VLLM_MOE_DP_CHUNK_SIZE
    enable_moe_dp_chunk: bool = True  # from VLLM_ENABLE_MOE_DP_CHUNK
```

#### 1.7 Extend `SchedulerConfig`

```python
@dataclass
class SchedulerConfig:
    # Existing fields...
    
    # New fields (migrated from envvars)
    sleep_when_idle: bool = False  # from VLLM_SLEEP_WHEN_IDLE
```

### Phase 2: Backend Selection (v0.9.0)

Create a unified backend selection system.

```python
@dataclass
class BackendConfig:
    """Configuration for backend/kernel selection."""
    
    # Attention backends
    use_flashinfer_sampler: bool | None = None  # from VLLM_USE_FLASHINFER_SAMPLER
    
    # MOE backends
    flashinfer_moe_backend: Literal["throughput", "latency", "masked_gemm"] = "latency"  # from VLLM_FLASHINFER_MOE_BACKEND
    use_flashinfer_moe_fp16: bool = False  # from VLLM_USE_FLASHINFER_MOE_FP16
    use_flashinfer_moe_fp8: bool = False  # from VLLM_USE_FLASHINFER_MOE_FP8
    # ... other MOE backend flags
    
    # GEMM backends
    nvfp4_gemm_backend: str | None = None  # from VLLM_NVFP4_GEMM_BACKEND
    use_deep_gemm: bool = True  # from VLLM_USE_DEEP_GEMM
    moe_use_deep_gemm: bool = True  # from VLLM_MOE_USE_DEEP_GEMM
    # ... other GEMM backend flags
```

### Phase 3: Observability/Monitoring (v0.10.0)

```python
@dataclass
class ObservabilityConfig:
    # Existing fields...
    
    # New fields (migrated from envvars)
    log_stats_interval: float = 10.0  # from VLLM_LOG_STATS_INTERVAL
    log_batchsize_interval: float = -1  # from VLLM_LOG_BATCHSIZE_INTERVAL
    custom_scopes_for_profiling: bool = False  # from VLLM_CUSTOM_SCOPES_FOR_PROFILING
    nvtx_scopes_for_profiling: bool = False  # from VLLM_NVTX_SCOPES_FOR_PROFILING
```

### Environment Variable Backward Compatibility

For backward compatibility during migration:

```python
@dataclass
class CacheConfig:
    kv_cache_layout: Literal["NHD", "HND"] | None = None
    
    def __post_init__(self):
        # Fall back to envvar if not explicitly set
        if self.kv_cache_layout is None:
            if envs.VLLM_KV_CACHE_LAYOUT is not None:
                logger.warning(
                    "VLLM_KV_CACHE_LAYOUT environment variable is deprecated. "
                    "Please use --kv-cache-layout CLI argument or "
                    "CacheConfig(kv_cache_layout=...) instead."
                )
                self.kv_cache_layout = envs.VLLM_KV_CACHE_LAYOUT
```

### CLI Argument Support

Add corresponding CLI arguments via `EngineArgs`:

```python
class EngineArgs:
    # Add new arguments
    kv_cache_layout: Literal["NHD", "HND"] | None = None
    
    @staticmethod
    def add_cli_args(parser: FlexibleArgumentParser):
        # ...
        cache_group.add_argument(
            "--kv-cache-layout",
            type=str,
            choices=["NHD", "HND"],
            default=None,
            help="KV cache memory layout (default: backend-specific)"
        )
```

## Migration Strategy

### 1. Deprecation Period (2 releases)

- Add config fields alongside envvars
- Emit deprecation warnings when envvar is used
- Update documentation to show new config approach
- Keep envvar as fallback

### 2. Migration Tools

Provide a migration helper:

```python
# Tool to detect deprecated envvar usage
python -m vllm.tools.check_deprecated_envvars

# Output:
# Warning: VLLM_KV_CACHE_LAYOUT is deprecated. Use --kv-cache-layout instead.
# Warning: VLLM_SLEEP_WHEN_IDLE is deprecated. Use --sleep-when-idle instead.
```

### 3. Documentation Updates

- Create migration guide showing envvar → config mapping
- Update all examples to use new config approach
- Add prominent warnings in docs for deprecated envvars

### 4. Removal (After 2 releases)

- Remove envvar fallback logic
- Remove deprecated envvars from `envs.py`
- Keep only system-level envvars

## Detailed Design

### Precedence Order

```
CLI argument > Explicit config > Environment variable (deprecated) > Default value
```

### Example Usage

**Before (envvar-based):**
```bash
VLLM_KV_CACHE_LAYOUT=NHD \
VLLM_SLEEP_WHEN_IDLE=1 \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf
```

**After (config-based):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --kv-cache-layout NHD \
    --sleep-when-idle
```

**Python API:**
```python
# Before
import os
os.environ["VLLM_KV_CACHE_LAYOUT"] = "NHD"
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# After
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_cache_layout="NHD",
    sleep_when_idle=True
)
```

## Testing Strategy

### 1. Unit Tests

```python
def test_cache_config_precedence():
    """Test that config takes precedence over envvar."""
    # Set envvar
    os.environ["VLLM_KV_CACHE_LAYOUT"] = "HND"
    
    # Explicit config should override
    config = CacheConfig(kv_cache_layout="NHD")
    assert config.kv_cache_layout == "NHD"
    
    # Without explicit config, should use envvar (with warning)
    config = CacheConfig()
    assert config.kv_cache_layout == "HND"
```

### 2. Integration Tests

- Test that existing examples work with new config
- Verify backward compatibility with envvars
- Ensure CLI arguments work correctly

### 3. Deprecation Warnings Tests

```python
def test_envvar_deprecation_warning():
    """Test that deprecated envvar usage emits warning."""
    os.environ["VLLM_KV_CACHE_LAYOUT"] = "NHD"
    
    with pytest.warns(DeprecationWarning, match="VLLM_KV_CACHE_LAYOUT"):
        config = CacheConfig()
```

## Drawbacks and Alternatives

### Drawbacks

1. **Migration Effort**: Users must update their scripts/configs
2. **Temporary Complexity**: During transition, both methods exist
3. **Breaking Changes**: Eventually removes envvar support

### Alternatives Considered

#### Alternative 1: Keep Everything as Envvars
- **Pros**: No migration needed, no breaking changes
- **Cons**: Technical debt continues to grow, harder to maintain

#### Alternative 2: Create Parallel Config System
- **Pros**: Both can coexist indefinitely
- **Cons**: Doubles complexity, confusing for users

#### Alternative 3: Big Bang Migration
- **Pros**: Clean break, no transition period
- **Cons**: Very disruptive to users, high risk

**Chosen Approach**: Gradual migration with deprecation warnings (balances safety and progress)

## Environment Variables That Should NOT Migrate

The following should remain as environment variables:

1. **System Paths**: `VLLM_CACHE_ROOT`, `VLLM_NCCL_SO_PATH`, etc.
2. **Build Configuration**: `MAX_JOBS`, `CMAKE_BUILD_TYPE`, etc.
3. **Debug/Development**: `VLLM_TRACE_FUNCTION`, `VLLM_DEBUG_*`, etc.
4. **Logging Setup**: `VLLM_LOGGING_*` (process-level configuration)
5. **System Integration**: `CUDA_VISIBLE_DEVICES`, `LOCAL_RANK`, etc.

**Guideline**: If it's about *where* things are in the system or is purely for debugging/development, it stays as envvar. If it controls *what* or *how* vLLM behaves with user requests, it becomes config.

## Success Metrics

1. **Reduction in envvars**: 50+ envvars migrated to configs
2. **User Feedback**: Positive response on ease of configuration
3. **Test Coverage**: >90% coverage for new config paths
4. **Documentation**: Complete migration guide published
5. **Adoption**: >80% of examples updated to new approach

## Timeline

- **v0.8.0** (Month 1-2): Phase 1 implementation, deprecation warnings
- **v0.9.0** (Month 3-4): Phase 2 implementation, migration tooling
- **v0.10.0** (Month 5-6): Phase 3 implementation, full documentation
- **v0.11.0** (Month 7+): Remove deprecated envvars

## Open Questions

1. Should we support YAML/JSON config files for complex setups?
2. How to handle connector-specific configs more elegantly?
3. Should we auto-generate CLI args from dataclass fields?
4. How to handle platform-specific configs (ROCm, TPU, etc.)?

## References

- Issue: Environment variable proliferation discussion
- Related: Config system refactoring (#XXXX)
- Inspiration: [Python's dataclasses](https://docs.python.org/3/library/dataclasses.html)
- Inspiration: [Hydra configuration framework](https://hydra.cc/)

## Appendix: Complete Migration Mapping

See separate document: `docs/migration/envvars-to-config.md`

| Environment Variable | Config Object | Config Field | Version |
|---------------------|---------------|--------------|---------|
| `VLLM_KV_CACHE_LAYOUT` | `CacheConfig` | `kv_cache_layout` | v0.8.0 |
| `VLLM_SLEEP_WHEN_IDLE` | `SchedulerConfig` | `sleep_when_idle` | v0.8.0 |
| `VLLM_FLOAT32_MATMUL_PRECISION` | `ModelConfig` | `float32_matmul_precision` | v0.8.0 |
| ... | ... | ... | ... |
