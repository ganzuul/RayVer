#!/bin/bash
#SBATCH --job-name=ray-vllm-2nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --time=00:30:00
#SBATCH --partition=dev-g
#SBATCH --account=project_465002004
#SBATCH --output=logs/ray_diag_%j.out
#SBATCH --error=logs/ray_diag_%j.err

set -e

# --- LUMI Modules & AI Bindings ---
#module purge # but then load libfabric etc etc...
module load LUMI/24.03 partition/G
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
RAY="python3 -m ray.scripts.scripts"
RAY_TASK="python3 ray_prompt_gen.py" #  <---  PYTHON SCRIPT HERE

# --- Enhanced Debug and Logging ---
export VLLM_CONFIGURE_LOGGING=1
export VLLM_LOGGING_LEVEL=INFO
export NCCL_DEBUG=INFO
export RAY_LOGGING_LEVEL=WARNING
# export RAY_LOG_TO_STDERR=1
export RAY_DISABLE_IMPORT_WARNING=1
export RAY_DATA_DISABLE_PROGRESS_BARS=1
export RAY_DEDUP_LOGS=1

# --- NCCL, HIP, Binding ---
# export NCCL_IB_DISABLE=1 # should fallback to Ethernet
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB # or 3
#export NCCL_DMABUF_ENABLE=1 # Segfaults
export RCCL_MSCCL_FORCE_ENABLE=1
export RCCL_MSCCL_ENABLE_SINGLE_PROCESS=1
# export RCCL_MSCCLPP_ENABLE=1  # Cannot enable MSCCL++ on gfx90a:sramecc+:xnack- architecture
export NCCL_MIN_NCHANNELS=32
export HSA_FORCE_FINE_GRAIN_PCIE=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
#export NUMEXPR_MAX_THREADS=7

# AMD tuning
# export HIP_FORCE_DEV_KERNARG=1
export TORCH_BLAS_PREFER_HIPBLASLT=1
# export PYTORCH_TUNABLEOP_ENABLED=1 

# --- Path and App Params ---
SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT:-"project_465002004"} 
PROJECT="/project/${SLURM_JOB_ACCOUNT}"
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
FLASH="/flash/${SLURM_JOB_ACCOUNT}"
export OUTPUT_DIR="$SCRATCH/$USER/output/ray_outputs_job$SLURM_JOB_ID"
export LOGGING_DIR="$SCRATCH/$USER/logs"
mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"

# HF and Torch Cache
ACTUAL_HF_HOME_PATH="$SCRATCH/huggingface" 
export HF_HOME="$ACTUAL_HF_HOME_PATH"
export HF_TOKEN=$USER_HF_TOKEN # per hf model argreements also needed
# export HF_HUB_CACHE="$ACTUAL_HF_HOME_PATH/hub"
export TORCH_HOME="$SCRATCH/torch-cache"
export SHARED_MODEL_CACHE="$FLASH/hf-cache/hub" #for download_dir
export HF_HUB_CACHE=$SHARED_MODEL_CACHE

mkdir -p "$TORCH_HOME" "$ACTUAL_HF_HOME_PATH" "$HF_HUB_CACHE"

# DSR170B: Outlines on: 125Tok/s ; Outlines off: 250Tok/s ; xgrammar 230Tok/s
export MODEL_ID_OR_PATH_ENV="deepseek-ai/DeepSeek-R1-Distill-Llama-70B" 
# export MODEL_ID_OR_PATH_ENV="deepseek-ai/DeepSeek-V3-0324" # Might OOM
# export MODEL_ID_OR_PATH_ENV="meta-llama/Llama-3.2-1B-Instruct" # 1044Tok/s
# export MODEL_ID_OR_PATH_ENV="EleutherAI/gpt-neo-1.3B"
export PROMPTS_FILE_ENV="${PROJECT}/${USER}/prompts.json"
export TENSOR_PARALLEL_SIZE_ENV=8
export PIPELINE_PARALLEL_SIZE_ENV=${SLURM_NNODES}
export GPU_MEMORY_UTILIZATION_ENV=0.90 # Needs tweaking, .85-95 
export MODEL_DTYPE_ENV="bfloat16"
export MAX_NEW_TOKENS=1024
export MAX_MODEL_LEN_ENV=2048

# --- Ray Data / Generation Specific ENVs ---
export TEMPERATURE="0.7"
export TOP_P=".9"
export TOP_K="-1"
export RAY_DATA_BATCH_SIZE="8"

# --- vLLM Specific ENVs ---
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
# export VLLM_USE_RAY_SPMD_WORKER=0
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND="ray"
export VLLM_ATTENTION_BACKEND="ROCM_FLASH"


# --- Singularity Setup ---
CONTAINER_ROOT=$FLASH
# from docker://rocm/vllm-dev:nightly plus custom overlay
CONTAINER_PATH="--overlay ${CONTAINER_ROOT}/vllm_mpich_overlay.sif ${CONTAINER_ROOT}/vllm-dev_nightly.sif" 
# CONTAINER_PATH="--overlay ${CONTAINER_ROOT}/vllm_mpich_overlay.sif ${CONTAINER_ROOT}/vllm_mpich.sif" 
# CONTAINER_PATH="/appl/local/containers/tested-containers/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.0-dockerhash-2a550b31226f.sif"
SINGULARITY_EXEC_BASE="singularity exec --rocm ${CONTAINER_PATH}"

# --- Ray Cluster Setup (2 nodes) ---
nodes_str=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
mapfile -t nodes_array <<< "$nodes_str"
head_node_hostname=${nodes_array[0]}
worker_node_hostname=${nodes_array[1]} #TO-DO: hyperscale
RAY_PORT=${RAY_PORT:-6379}

# Get the IP address of the head node ## TO-DO try to clean this up since only RAY_ADDRESS needs this
raw_head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node_hostname" hostname --ip-address)
echo "DEBUG: Raw IP address for $head_node_hostname: $raw_head_node_ip"

head_node_ip=""
for ip_addr in $raw_head_node_ip; do
    if [[ $ip_addr =~ ^10\. ]] || [[ $ip_addr =~ ^172\.(1[6-9]|2[0-9]|3[0-1])\. ]] || [[ $ip_addr =~ ^192\.168\. ]]; then
        head_node_ip=$ip_addr # Prefer private IPv4
        break
    fi
done
if [ -z "$head_node_ip" ]; then # If no private IPv4 found, try any IPv4
    for ip_addr in $raw_head_node_ip; do
        if [[ $ip_addr =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            head_node_ip=$ip_addr
            break
        fi
    done
fi
if [ -z "$head_node_ip" ]; then # Fallback to the first IP if no IPv4 was clearly identified
    head_node_ip=$(echo "$raw_head_node_ip" | awk '{print $1}')
    echo "WARNING: Could not definitively identify an IPv4 address; using first reported IP: $head_node_ip"
fi

export RAY_ADDRESS="${head_node_ip}:${RAY_PORT}"

RAY_NODE_CPUS=$SLURM_CPUS_PER_TASK
RAY_NODE_GPUS=$SLURM_GPUS_PER_NODE
RAY_NODE_SETUP_CMDS="export RAY_DISABLE_MEMORY_MONITOR=1; \
export RAY_TMPDIR=/tmp/ray; \
export MIOPEN_USER_DB_PATH=/tmp/$(whoami)-miopen-cache-$SLURM_JOBID-\$SLURM_NODEID; mkdir -p \$MIOPEN_USER_DB_PATH; export MIOPEN_CUSTOM_CACHE_DIR=\$MIOPEN_USER_DB_PATH; \
export LD_LIBRARY_PATH=/usr/lib64/mpi/gcc/mpich/lib64:\$LD_LIBRARY_PATH &&
ldd /opt/aws-ofi-rccl/librccl-net.so | grep fabric &&
ls /opt/cray/libfabric/1.15.2.0/lib64/libfabric.so.1 &&
ulimit -n 65536; \
ulimit -u 32768; \
echo 'Node setup complete, Ray version:'; \
python3 -c 'import ray; print(ray.__version__)';"
#"echo 'GPU visibility:'; python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"Device count: {torch.cuda.device_count()}\")' || echo 'PyTorch GPU check failed';"

# --- Explicitly propagate env vars into Singularity for the Python script ---
export SINGULARITYENV_HF_HOME_ENV=$ACTUAL_HF_HOME_PATH
export SINGULARITYENV_SHARED_MODEL_CACHE_ENV=$SHARED_MODEL_CACHE
export SINGULARITYENV_OUTPUT_DIR_ENV=$OUTPUT_DIR # Python script will use this
export SINGULARITYENV_LOGGING_DIR_ENV=$LOGGING_DIR # Python script will use this
export SINGULARITYENV_MAX_NEW_TOKENS=$MAX_NEW_TOKENS
export SINGULARITYENV_HF_TOKEN=$HF_TOKEN
export SINGULARITYENV_MODEL_ID_OR_PATH_ENV=$MODEL_ID_OR_PATH_ENV
export SINGULARITYENV_PROMPTS_FILE_ENV=$PROMPTS_FILE_ENV
export SINGULARITYENV_TENSOR_PARALLEL_SIZE_ENV=$TENSOR_PARALLEL_SIZE_ENV
export SINGULARITYENV_PIPELINE_PARALLEL_SIZE_ENV=$PIPELINE_PARALLEL_SIZE_ENV
export SINGULARITYENV_GPU_MEMORY_UTILIZATION_ENV=$GPU_MEMORY_UTILIZATION_ENV
export SINGULARITYENV_MODEL_DTYPE_ENV=$MODEL_DTYPE_ENV
export SINGULARITYENV_MAX_MODEL_LEN_ENV=$MAX_MODEL_LEN_ENV

# AMD tuning
# export SINGULARITYENV_HIP_FORCE_DEV_KERNARG=$HIP_FORCE_DEV_KERNARG
# export SINGULARITYENV_TORCH_BLAS_PREFER_HIPBLASLT=$TORCH_BLAS_PREFER_HIPBLASLT
# export SINGULARITYENV_PYTORCH_TUNABLEOP_ENABLED=$PYTORCH_TUNABLEOP_ENABLED

# --- Ray Data / Generation Specific ENVs for Singularity ---
export SINGULARITYENV_TEMPERATURE=$TEMPERATURE
export SINGULARITYENV_TOP_P=$TOP_P
export SINGULARITYENV_TOP_K=$TOP_K
export SINGULARITYENV_RAY_DATA_BATCH_SIZE=$RAY_DATA_BATCH_SIZE

# Ray configuration
export SINGULARITYENV_RAY_ADDRESS=$RAY_ADDRESS
# export SINGULARITYENV_RAY_LOGGING_LEVEL="$RAY_LOGGING_LEVEL"
# export SINGULARITYENV_RAY_LOG_TO_STDERR="$RAY_LOG_TO_STDERR"
export SINGULARITYENV_RAY_DISABLE_IMPORT_WARNING=$RAY_DISABLE_IMPORT_WARNING
export SINGULARITYENV_RAY_DATA_DISABLE_PROGRESS_BARS=$RAY_DATA_DISABLE_PROGRESS_BARS
export SINGULARITYENV_RAY_DEDUP_LOGS=$RAY_DEDUP_LOGS
export SINGULARITYENV_RAY_DATA_DISABLE_PROGRESS_BARS=$RAY_DATA_DISABLE_PROGRESS_BARS
# Hardware configuration
export SINGULARITYENV_NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export SINGULARITYENV_NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL
#export SINGULARITYENV_NCCL_DMABUF_ENABLE=$NCCL_DMABUF_ENABLE
export SINGULARITYENV_RCCL_MSCCL_FORCE_ENABLE=$RCCL_MSCCL_FORCE_ENABLE
export SINGULARITYENV_RCCL_MSCCL_ENABLE_SINGLE_PROCESS=$RCCL_MSCCL_ENABLE_SINGLE_PROCESS
# export SINGULARITYENV_RCCL_MSCCLPP_ENABLE=$RCCL_MSCCLPP_ENABLE
export SINGULARITYENV_NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS
export SINGULARITYENV_NCCL_DEBUG=$NCCL_DEBUG
export SINGULARITYENV_HSA_FORCE_FINE_GRAIN_PCIE=$HSA_FORCE_FINE_GRAIN_PCIE

# vLLM configuration
export SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD=$VLLM_WORKER_MULTIPROC_METHOD
export SINGULARITYENV_VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL
# export SINGULARITYENV_VLLM_USE_RAY_SPMD_WORKER=$VLLM_USE_RAY_SPMD_WORKER
export SINGULARITYENV_VLLM_DISTRIBUTED_EXECUTOR_BACKEND=$VLLM_DISTRIBUTED_EXECUTOR_BACKEND
export SINGULARITYENV_VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND
export SINGULARITYENV_VLLM_CONFIGURE_LOGGING=$VLLM_CONFIGURE_LOGGING

# Start Ray Head
echo "DEBUG: Starting Ray head node on $head_node_hostname"
srun --nodes=1 --ntasks=1 -w "$head_node_hostname" --jobid $SLURM_JOBID --cpus-per-task="$RAY_NODE_CPUS" --gpus="$RAY_NODE_GPUS" \
    $SINGULARITY_EXEC_BASE bash -c "${RAY_NODE_SETUP_CMDS} \
            $RAY start --head --port=${RAY_PORT} --num-cpus=${RAY_NODE_CPUS} --num-gpus=${RAY_NODE_GPUS} --block --disable-usage-stats  --include-dashboard=False  --verbose & \
            sleep 20;
            echo 'Python client environment:'; \
            env | grep -E 'RAY_|SLURM_|NCCL_|HSA_'; \
            env | grep -E 'SINGULARITY'; \
            pip list; \
            uname -a; \
            cat /etc/os-release; \
            $RAY status; \
            $RAY_TASK" &
RAY_HEAD_PID=$!
echo "DEBUG: Ray head process started with PID $RAY_HEAD_PID. Waiting for initialization..."
sleep 5

# Start Ray Worker  ## TO-DO: more nodes
echo "DEBUG: Starting Ray worker on $worker_node_hostname, connecting to $head_node_hostname:${RAY_PORT}"
srun --nodes=1 --ntasks=1 --exclude=$head_node_hostname --jobid $SLURM_JOBID --cpus-per-task="$RAY_NODE_CPUS" --gpus="$RAY_NODE_GPUS" \
    $SINGULARITY_EXEC_BASE bash -c "${RAY_NODE_SETUP_CMDS} \
            $RAY start --address=$head_node_hostname:${RAY_PORT}  --num-cpus=${RAY_NODE_CPUS} --num-gpus=${RAY_NODE_GPUS}  --block --disable-usage-stats" &
echo "DEBUG: Ray worker srun command issued. Waiting for it to connect..."
sleep 3000 # TO-DO: Exit gracefully

# --- Cleanup ---
# Since ray start is now blocking within srun, direct PID waiting might not be applicable
# if srun itself is the process we wait for.
# However, if ray start daemonizes *within* the srun, then RAY_HEAD_PID might be the srun PID
# which would have exited.
# The job's main purpose is diagnostic, so we'll proceed to ray stop.
# Original wait logic:
# if ps -p $RAY_HEAD_PID > /dev/null; then
#     echo "Waiting for Ray head process (PID $RAY_HEAD_PID) to complete..."
#     wait $RAY_HEAD_PID
# else
#     echo "Ray head process (PID $RAY_HEAD_PID) already exited or not found."
# fi
# Given the change to foreground, the script will wait for srun to complete.
# If the python script signals completion or we rely on time limits, that's the main control flow.

echo "Attempting to stop Ray cluster..."
# srun --nodes=1 --ntasks=1 -w "$head_node_hostname" --jobid $SLURM_JOBID \
#     bash -c "${RAY_NODE_SETUP_CMDS}$RAY stop" &
scancel -n $SLURM_JOB_ID # Not graceful
echo "Script finished."
