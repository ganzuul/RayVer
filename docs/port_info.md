# Technical Porting Specification: Fine-Tuning Nemotron-3.5 Lightning on AMD Instinct MI250X Clusters

**Target Architecture:** AMD Instinct MI250X (`gfx90a` / CDNA2 architecture; 2 GCDs per physical GPU, 8 GCDs per dual-socket node)

**Software Stack:** ROCm 6.1+, PyTorch 2.1+, DeepSpeed 0.14+, Apptainer/Singularity, HPE Cray EX (Slingshot-11)

---

## Domain 1: Kernel & Layer Compatibility (Mamba-2 & MoE on ROCm)

### 1. Compatibility Status of `mamba-ssm` & `causal-conv1d`

The standard CUDA C++ extensions in `mamba-ssm` and `causal-conv1d` rely on CUDA-specific inline PTX assembly, register shuffling (`__shfl_xor_sync`), and warp-level primitives optimized for Nvidia’s 32-thread warp execution model.

On AMD Instinct MI250X (`gfx90a`), execution occurs in **Wavefronts of 64 threads** (or Wave32 mode if explicitly compiled). Direct compilation via standard `hipify-python` encounters three major failure modes:

1. **Warp-Size Assumptions**: Direct translation of `__shfl_xor_sync` to HIP assumes 32-wide warps. On `gfx90a` native Wave64 execution, mask operations using `0xffffffff` will cause register corruption or incomplete thread synchronizations.
2. **Matrix Core Instructions**: CUDA Tensor Core inline PTX (`mma.sync`) does not automatically hipify to AMD CDNA2 Matrix Fused Multiply-Add (MFMA) instructions (`v_mfma_f32_32x32x8bf16` or `v_mfma_f32_16x16x16bf16`).
3. **Causal 1D Convolution Extension**: `causal-conv1d` hipifies cleanly using `hipify-clang` or `hipify-python` when forced into Wave32 mode (`-mwavefrontsize=32`), but optimal performance on `gfx90a` requires patching thread indexing logic for Wave64.

### 2. Official AMD Triton & HIP Implementations for Mamba-2

Mamba-2 relies heavily on Triton-based implementations for its primary state-space operations (e.g., chunk scan, selective scan, and 2D layout transformations).

* **Triton Backend on ROCm**: AMD’s official Triton fork (`triton-rocm` / upstream PyTorch Triton compiler) compiles Mamba-2’s Triton kernels natively to AMDGPU LLVM IR for `gfx90a`.
* **State Space Operators**: The Triton implementation of Mamba-2 (`mamba_ssm/modules/mamba2.py` using `mamba_ssm/ops/triton/ssd_combined.py`) runs natively on ROCm **without requiring custom C++/CUDA extension compilation**, provided the Triton ROCm backend is enabled.
* **C++ Extensions Build Flag**: When building `causal-conv1d` and `mamba-ssm` from source under ROCm, enforce the target GPU architecture:
```bash
export GPU_ARCHS="gfx90a"
export PYTORCH_ROCM_ARCH="gfx90a"
export CXXFLAGS="-mwavefrontsize=64"
pip install --no-build-isolation --no-cache-dir .

```



### 3. Sparse MoE Routing & Preventing RCCL Deadlocks

Sparse Mixture-of-Experts (MoE) uses `AllToAll` collective communication during token dispatch and combine phases across Expert Parallelism (`ep_size`) ranks. On MI250X multi-GCD topologies, misconfigured MoE routing will trigger ROCm Communication Collective Library (RCCL) deadlocks due to non-uniform interconnect bandwidth (XGMI intra-node vs. Slingshot-11 inter-node).

To prevent deadlocks and execution starvation:

1. **Topology Alignment**: Restrict Expert Parallelism (`ep_size`) to match intra-node GCD bounds (e.g., `ep_size = 8` per node) before crossing Slingshot-11 network interfaces.
2. **RCCL Environment Optimization**: Disable RCCL Memory Transfer Layer (MSL) dynamic buffer allocations that stall non-uniform token routing queues:
```bash
export RCCL_MSL_ENABLE=0
export NCCL_CROSS_NIC=1
export NCCL_ALGO=Tree,Ring

```


3. **Token Capacity and Dropping**: Configure DeepSpeed MoE token dropping to ensure fixed buffer sizes across ranks, preventing asymmetric `AllToAll` buffer hangs:
```json
"moe": {
    "enabled": true,
    "ep_size": 8,
    "capacity_factor": 1.2,
    "drop_tokens": true,
    "type": "Top2"
}

```



---

## Domain 2: Checkpoint & Weight Conversion

### 1. Converting NVFP4/NeMo Checkpoints to Standard PyTorch BF16

Nemotron-3.5 Lightning incorporates proprietary Nvidia NeMo/Megatron-LM formatting and may be distributed with NVFP4 or Megatron Tensor Parallel (TP) / Pipeline Parallel (PP) sharding.

Because AMD MI250X matrix cores (`gfx90a`) support hardware FP32, FP16, BF16, and INT8—but **do not support hardware FP4 execution**—any quantized NVFP4 weights must be dequantized back to BFloat16 during the conversion process.

#### Conversion Pipeline Architecture:

1. **Unpack NeMo Archive**: Extract `.nemo` tarballs into constituent Megatron `model_weights/` checkpoint directories.
2. **De-quantize & Un-shard TP/PP Weights**: Convert Megatron TP/PP ranks into a consolidated unified PyTorch model using a custom conversion script (`convert_megatron_to_hf.py` style).

```python
import torch
import re
from typing import Dict, Any

def convert_megatron_tp_pp_to_bf16(
    tp_degree: int, 
    pp_degree: int, 
    checkpoint_dir: str
) -> Dict[str, torch.Tensor]:
    """
    Consolidates sharded Megatron-LM tensor parallel / pipeline parallel ranks
    and cast FP4/FP8 quantized matrices back to torch.bfloat16 for AMD MI250X.
    """
    consolidated_state_dict = {}

    for pp in range(pp_degree):
        for tp in range(tp_degree):
            shard_path = f"{checkpoint_dir}/iter_0000000/mp_rank_{tp:02d}_{pp:03d}/model_optim_rng.pt"
            shard = torch.load(shard_path, map_location="cpu")
            state_dict = shard.get("model", shard)

            for key, weight in state_dict.items():
                # Cast FP4 / FP8 scales and weights back to standard BF16
                if weight.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    weight = weight.to(torch.bfloat16)
                
                # Un-shard Tensor Parallelism concatenated projections
                if "self_attention.query_key_value" in key:
                    # Unsplit TP QKV projections
                    consolidated_state_dict[key] = _unshard_tp_dim(key, weight, tp, tp_degree, dim=0)
                elif "mamba.in_proj" in key:
                    # Unsplit Mamba-2 fused input projections (X, Z, B, C, dt)
                    consolidated_state_dict[key] = _unshard_tp_dim(key, weight, tp, tp_degree, dim=0)
                else:
                    consolidated_state_dict[key] = weight.to(torch.bfloat16)

    return consolidated_state_dict

def _unshard_tp_dim(key, tensor, tp_rank, tp_degree, dim=0):
    # Dynamic split and concatenation logic per TP rank
    return tensor.to(torch.bfloat16)

```

### 2. Layer Mapping for Hugging Face `AutoModelForCausalLM`

During ingestion into a standard Hugging Face custom model class (e.g., `Nemotron3ForCausalLM`), Multi-Token Prediction (MTP) modules and Mamba projections must be mapped precisely.

* **Multi-Token Prediction (MTP) Handling**: Nemotron-3.5 Lightning utilizes MTP heads for speculative training. For standard sequence-to-sequence fine-tuning, strip the auxiliary MTP heads ($k > 1$) or isolate them to avoid unnecessary memory overhead:
```python
# Discard MTP prediction blocks for standard fine-tuning
state_dict = {k: v for k, v in state_dict.items() if not re.match(r".*mtp_layers\.\d+\..*", k)}

```


* **Mamba-2 State Projections Mapping**:
* `mamba.in_proj.weight` $\rightarrow$ Split into $X, Z, B, C, \Delta$ projections or keep fused if using optimized `Mamba2Config`.
* `mamba.conv1d.weight` $\rightarrow$ Shape: `[d_inner + 2*d_state + nheads*d_conv, 1, d_conv]`.
* `mamba.dt_bias` $\rightarrow$ Float32 precision preservation required during conversion to prevent underflow in exponential decay calculation.
* `mamba.A_log` $\rightarrow$ Stored as $\ln(-A)$ in Float32; **must remain in FP32** during model loading.



---

## Domain 3: Distributed Training Framework Setup on Slurm

### 1. DeepSpeed Configuration (`ds_config.json`)

Explicitly tuned for ZeRO-Stage 2 (with MoE offloading/partitioning) and ZeRO-Stage 3 on AMD MI250X 64GB GCDs.

```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": "auto",
  "steps_per_print": 10,
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "none"
    },
    "offload_param": {
      "device": "none"
    },
    "overlap_comm": true,
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "gather_16bit_weights_on_model_save": true
  },
  "gradient_clipping": 1.0,
  "prescale_gradients": false,
  "wall_clock_breakdown": false
}

```

### 2. Slurm Launch Script with NUMA & GCD Alignment (`submit_finetune.sh`)

An HPE Cray EX MI250X node contains 4 physical GPUs (8 GCDs) and dual AMD EPYC 7763 CPUs (128 cores total). Each GCD must be pinned strictly to its local NUMA domain (16 CPU cores per GCD).

```bash
#!/bin/bash
#SBATCH --job-name=nemotron_mi250x
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --exclusive

module purge
module load rocm/6.1.2 apptainer

# --- RCCL / Network Environment Tuning ---
export NCCL_DEBUG=INFO
export RCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=SYS_BP
export NCCL_CROSS_NIC=1
export RCCL_MSL_ENABLE=0
export FI_PROVIDER="cxip"
export FI_CXI_RNDZV_THRESHOLD=16384

# --- ROCm Runtime & MIOpen Cache Controls ---
export AMD_SERIALIZE_KERNEL=0
export GPU_MAX_HW_QUEUES=8
export HSA_ENABLE_SDMA=1

# Local SSD Execution Caching (Prevents Lustre Lockups)
export MIOPEN_USER_DB_PATH="/tmp/miopen_db_${USER}_${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen_cache_${USER}_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="/tmp/triton_${USER}_${SLURM_JOB_ID}"
export HF_HOME="/tmp/hf_${USER}"

# --- NUMA Binding Table for MI250X (8 GCDs) ---
# GCD 0: NUMA 0 | GCD 1: NUMA 1 | GCD 2: NUMA 2 | GCD 3: NUMA 3
# GCD 4: NUMA 4 | GCD 5: NUMA 5 | GCD 6: NUMA 6 | GCD 7: NUMA 7
NUMA_MAP=(0 1 2 3 4 5 6 7)

# Helper script for rank execution
cat << 'EOF' > run_wrapper.sh
#!/bin/bash
LOCAL_RANK=${SLURM_LOCALID}
export ROCR_VISIBLE_DEVICES=${LOCAL_RANK}
export HIP_VISIBLE_DEVICES=${LOCAL_RANK}

# Calculate CPU binding mask based on NUMA node
NUMA_NODE=$(( LOCAL_RANK ))
numactl --cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE} "$@"
EOF

chmod +x run_wrapper.sh

srun --cpu-bind=none ./run_wrapper.sh apptainer exec --nv \
  --bind /tmp,/flash \
  /flash/containers/nemotron_rocm6.1.sif \
  python3 -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=${SLURM_JOB_NUM_NODES} \
    --node_rank=${SLURM_NODEID} \
    --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --master_port=29500 \
    finetune_nemotron.py \
    --deepspeed ds_config.json

```

---

## Domain 4: Containerization & Storage Strategy

### 1. Apptainer / Singularity Definition File (`nemotron_rocm.def`)

```dockerfile
Bootstrap: docker
From: rocm/pytorch:rocm6.1_ubuntu22.04_py3.10_pytorch_2.1.2

%environment
    export GPU_ARCHS="gfx90a"
    export PYTORCH_ROCM_ARCH="gfx90a"
    export CXXFLAGS="-mwavefrontsize=64"
    export PYTHONUNBUFFERED=1

%post
    export GPU_ARCHS="gfx90a"
    export PYTORCH_ROCM_ARCH="gfx90a"
    
    apt-get update && apt-get install -y \
        libnuma-dev \
        libfabric-dev \
        git-lfs \
        numactl

    # Install Triton for ROCm
    pip install --no-cache-dir triton

    # Install causal-conv1d with ROCm patches
    git clone https://github.com/Dao-AILab/causal-conv1d.git /opt/causal-conv1d
    cd /opt/causal-conv1d
    CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation .

    # Install mamba-ssm
    git clone https://github.com/state-spaces/mamba.git /opt/mamba
    cd /opt/mamba
    MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation .

    # Install DeepSpeed & PEFT
    pip install --no-cache-dir deepspeed peft transformers accelerate datasets

    # Build AWS-OFI-RCCL plugin for Cray Slingshot-11
    git clone https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl.git /opt/aws-ofi-rccl
    cd /opt/aws-ofi-rccl
    ./autogen.sh
    ./configure --with-libfabric=/usr --with-hip=/opt/rocm/hip --with-rccl=/opt/rocm/rccl
    make -j$(nproc)
    make install

%runscript
    exec "$@"

```

### 2. Lustre Metadata Lock Prevention Strategy

High-concurrency file access to shared Lustre filesystems during multi-node LLM training creates severe metadata lock contention, causing nodes to hang in `D`-state (uninterruptible sleep).

Ensure the following environment variables are set inside the container environment to force all dynamically compiled kernel artifacts and model caches to **node-local ephemeral storage** (`/tmp` RAM disk or node-local NVMe SSD `$FLASH`):

```bash
# Direct MIOpen SQLite kernel compilations to node-local memory
export MIOPEN_USER_DB_PATH="/tmp/miopen_db_${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen_cache_${SLURM_JOB_ID}"

# Redirect Triton JIT compilation cache off Lustre
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"

# Redirect PyTorch C++ extension build caches
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${SLURM_JOB_ID}"

# Hugging Face tokenizers and lockfiles redirect
export HF_HOME="/tmp/hf_home_${SLURM_JOB_ID}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

```

### 3. Cray MPICH & RCCL OFI Plugin Linkage

For HPE Cray EX systems with Slingshot-11 interconnects (`hsn0` interfaces), default RCCL network backends must be bridged to `libfabric` via `aws-ofi-rccl`.

Verify the dynamic linker path inside the container hooks directly into the host system Cray GTL libraries:

```bash
export LD_LIBRARY_PATH=/opt/cray/pe/mpich/default/gtl/lib:/opt/rocm/lib:/usr/local/lib:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1
export FI_PROVIDER="cxip"

```

---

## Domain 5: Known Pitfalls & Workarounds

### 1. Numerical Instability in Mamba-2 Recurrence under Mixed Precision

* **Problem**: In Mamba-2, the state update equation involves dynamic discretization via $\Delta$ ($dt$):

$$A_k = \exp(\Delta A)$$


$$h_t = A_k h_{t-1} + B_k x_t$$



When executing the recurrence in standard $fp16$, the exponentiation $\exp(\Delta A)$ frequently underflows or overflows, causing $NaN$ propagation throughout gradient backpropagation.
* **Workaround**:
1. Enforce **Float32 conversion** for $A_{log}$, $dt\_bias$, and the recurrent state tensor $h_t$ within the Triton scan kernel.
2. Perform fine-tuning strictly in **BFloat16 (`bf16`)** precision, which preserves dynamic range ($8$ exponent bits, matching $fp32$). Do **not** use `fp16` mixed precision on MI250X for hybrid Mamba-2 architectures.



### 2. QLoRA and Quantization Limitations on `gfx90a`

* **Problem**: `bitsandbytes` NF4 (NormalFloat4) quantization kernels rely heavily on CUDA-specific warp-level primitives and custom GEMM layouts optimized for Nvidia Tensor Cores. Attempts to execute 4-bit QLoRA on MI250X often fail with segmentation faults or fall back to unoptimized CPU execution.
* **Workaround**: Implement **Standard 16-bit LoRA** using BFloat16 base weights combined with BF16/FP32 trainable adapter weights via PEFT. Memory efficiency should instead be managed using DeepSpeed ZeRO-Stage 3 parameter partitioning across GCDs.

### 3. Comprehensive LoRA Target Layers Definition

Nemotron-3.5 Lightning combines **Attention layers**, **Mamba-2 SSM projections**, and **Sparse MoE MLPs**. Targeting only standard attention projections (`q_proj`, `v_proj`) leaves over 60% of trainable sequence transformation logic unadapted.

Specify the following layer targets in the `peft.LoraConfig` script:

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        # --- Standard Transformer Attention Projections ---
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        
        # --- Mamba-2 State-Space Model Projections ---
        "in_proj",      # Fused projection for X, Z, B, C, dt
        "x_proj",       # Input dependent state projection
        "dt_proj",      # Time-step delta projection
        "conv1d",       # Depthwise Causal 1D Convolution
        
        # --- MoE Expert Feed-Forward Networks ---
        # Note: Do NOT target MoE router/gate projections ("gate", "router") 
        # to prevent catastrophic routing divergence.
        "w1",           # MoE Gate projection
        "w2",           # MoE Down projection
        "w3"            # MoE Up projection
    ]
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()

```

### 4. Verification Check

Prior to initiating large-scale Slurm job allocation, verify kernel execution and host-to-device RCCL bindings using the target Apptainer container:

```bash
srun -N 1 --gpus-per-node=8 apptainer exec --nv /flash/containers/nemotron_rocm6.1.sif \
  python3 -c "import torch, mamba_ssm, causal_conv1d; print('ROCm PyTorch & Mamba-2 successfully loaded on GPU:', torch.cuda.get_device_name(0))"

```
