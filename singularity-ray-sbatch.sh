#!/bin/bash
#SBATCH --job-name=ray-vllm-2nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --time=00:20:00
#SBATCH --partition=dev-g
#SBATCH --account=project_465002004
#SBATCH --output=logs/ray_diag_%j.out
#SBATCH --error=logs/ray_diag_%j.err

set -x # For debugging

# --- LUMI Modules & AI Bindings ---
#module purge # but then load libfabric etc etc...
#module use /appl/local/csc/modulefiles/
#module load pytorch/2.5 ## too old, we get 2.7 from custom container 
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
RAY="python3 -m ray.scripts.scripts"
RAY_TASK="python3 ray_prompt_gen.py" #  <---  PYTHON SCRIPT HERE

# --- NCCL, HIP, Binding ---
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB # or 3
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NUMEXPR_MAX_THREADS=128 

# --- Path and App Params ---
SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT:-"project_465002004"} 
PROJECT="/project/${SLURM_JOB_ACCOUNT}"
SCRATCH="/scratch/${SLURM_JOB_ACCOUNT}"
FLASH="/flash/${SLURM_JOB_ACCOUNT}"
export OUTPUT_DIR="$SCRATCH/$USER/output/ray_outputs_job$SLURM_JOB_ID"
export LOGGING_DIR="$SCRATCH/$USER/logs"
mkdir -p "$OUTPUT_DIR" "$LOGGING_DIR"

# HF and Torch Cache
ACTUAL_HF_HOME_PATH="$SCRATCH/huggingface/" 
export HF_HOME="$ACTUAL_HF_HOME_PATH"
export HF_TOKEN=USER_HF_TOKEN # per hf model argreements also needed
export HF_HUB_CACHE="$ACTUAL_HF_HOME_PATH/hub/"
export TORCH_HOME="$SCRATCH/torch-cache"
mkdir -p "$TORCH_HOME" "$ACTUAL_HF_HOME_PATH" "$HF_HUB_CACHE"

export MODEL_ID_OR_PATH_ENV="deepseek-ai/DeepSeek-R1-Distill-Llama-70B" # 125Tok/s, no SPMD
#export MODEL_ID_OR_PATH_ENV="deepseek-ai/DeepSeek-V3-0324" # Might OOM
export PROMPTS_FILE_ENV="${PROJECT}/${USER}/prompts.json"
export TENSOR_PARALLEL_SIZE_ENV=8
export PIPELINE_PARALLEL_SIZE_ENV=${SLURM_NNODES}
export GPU_MEMORY_UTILIZATION_ENV=0.85 # Needs tweaking, .85-95 
export MODEL_DTYPE_ENV="bfloat16"
export MAX_NEW_TOKENS=1024
export MAX_MODEL_LEN_ENV=2048

# --- Ray Data / Generation Specific ENVs ---
export TEMPERATURE="0.5" # Default temperature, used by prompt_gen.py
export TOP_P="1.0"       # Default top_p, used by prompt_gen.py
export TOP_K="-1"        # Default top_k, used by prompt_gen.py
export RAY_DATA_BATCH_SIZE="16" # Default batch size for Ray Data processing

# --- vLLM Specific ENVs ---
export VLLM_USE_RAY_SPMD_WORKER=1                       # Brittle, maybe not needed
export VLLM_USE_RAY_COMPILED_DAG=1                      # Brittle, maybe not needed
export VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL=1         # Brittle, maybe not needed
export VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=1         # Brittle, maybe not needed
export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE="nccl"    # Brittle, maybe not needed
export VLLM_USE_TRITON_FLASH_ATTN=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_PAGED_ATTN=1
export VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON=1
export VLLM_ATTENTION_BACKEND="ROCM_FLASH"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"

# --- Singularity Setup ---
CONTAINER_ROOT=$FLASH
# docker://rocm/vllm-dev:nightly plus custom overlay
CONTAINER_PATH="--overlay ${CONTAINER_ROOT}/vllm_mpich_overlay.sif ${CONTAINER_ROOT}/vllm_mpich.sif" 
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

export RAY_CLIENT_ADDRESS="${head_node_ip}:${RAY_PORT}"

RAY_NODE_CPUS=$SLURM_CPUS_PER_TASK
RAY_NODE_GPUS=$SLURM_GPUS_PER_NODE
RAY_NODE_SETUP_CMDS="export RAY_DISABLE_MEMORY_MONITOR=1; \
ulimit -n 65536; "

# --- Propagate ALL necessary env vars into Singularity for the Python script ---
export SINGULARITYENV_HF_HOME_ENV="$ACTUAL_HF_HOME_PATH"
export SINGULARITYENV_SHARED_MODEL_CACHE_ENV="$ACTUAL_HF_HOME_PATH"
export SINGULARITYENV_OUTPUT_DIR_ENV="$OUTPUT_DIR" # Python script will use this
export SINGULARITYENV_LOGGING_DIR_ENV="$LOGGING_DIR" # Python script will use this
export SINGULARITYENV_MAX_NEW_TOKENS=$MAX_NEW_TOKENS
export SINGULARITYENV_HF_TOKEN=$HF_TOKEN
export SINGULARITYENV_MODEL_ID_OR_PATH_ENV=$MODEL_ID_OR_PATH_ENV
export SINGULARITYENV_PROMPTS_FILE_ENV=$PROMPTS_FILE_ENV
export SINGULARITYENV_TENSOR_PARALLEL_SIZE_ENV=$TENSOR_PARALLEL_SIZE_ENV
export SINGULARITYENV_PIPELINE_PARALLEL_SIZE_ENV=$PIPELINE_PARALLEL_SIZE_ENV
export SINGULARITYENV_GPU_MEMORY_UTILIZATION_ENV=$GPU_MEMORY_UTILIZATION_ENV
export SINGULARITYENV_MODEL_DTYPE_ENV=$MODEL_DTYPE_ENV
export SINGULARITYENV_MAX_MODEL_LEN_ENV=$MAX_MODEL_LEN_ENV

# --- Ray Data / Generation Specific ENVs for Singularity ---
export SINGULARITYENV_TEMPERATURE=$TEMPERATURE
export SINGULARITYENV_TOP_P=$TOP_P
export SINGULARITYENV_TOP_K=$TOP_K
export SINGULARITYENV_RAY_DATA_BATCH_SIZE=$RAY_DATA_BATCH_SIZE
export SINGULARITYENV_RAY_ADDRESS="${RAY_CLIENT_ADDRESS}" # For the client script to connect

# Propagate for Ray workers (NCCL, HSA etc.)
export SINGULARITYENV_NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export SINGULARITYENV_NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL
export SINGULARITYENV_HSA_FORCE_FINE_GRAIN_PCIE=$HSA_FORCE_FINE_GRAIN_PCIE
export SINGULARITYENV_PYTORCH_HIP_ALLOC_CONF=$PYTORCH_HIP_ALLOC_CONF

# vLLM
export SINGULARITYENV_VLLM_USE_RAY_SPMD_WORKER=$VLLM_USE_RAY_SPMD_WORKER
export SINGULARITYENV_VLLM_USE_RAY_COMPILED_DAG=$VLLM_USE_RAY_COMPILED_DAG
export SINGULARITYENV_VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL=$VLLM_USE_RAY_COMPILED_DAG_NCCL_CHANNEL
export SINGULARITYENV_VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=$VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM
export SINGULARITYENV_VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=$VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE
export SINGULARITYENV_VLLM_USE_TRITON_FLASH_ATTN=$VLLM_USE_TRITON_FLASH_ATTN
export SINGULARITYENV_VLLM_ROCM_USE_AITER=$VLLM_ROCM_USE_AITER
export SINGULARITYENV_VLLM_ROCM_USE_AITER_PAGED_ATTN=$VLLM_ROCM_USE_AITER_PAGED_ATTN
export SINGULARITYENV_VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON=$VLLM_ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON
export SINGULARITYENV_VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND
export SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD=$VLLM_WORKER_MULTIPROC_METHOD

# Start Ray Head
echo "DEBUG: Starting Ray head node on $head_node_hostname"
srun --nodes=1 --ntasks=1 -w "$head_node_hostname" --jobid $SLURM_JOBID --cpus-per-task="$RAY_NODE_CPUS" --gpus="$RAY_NODE_GPUS" \
    $SINGULARITY_EXEC_BASE bash -c "${RAY_NODE_SETUP_CMDS} \
            echo 'Which Ray: '; which ray; \
            $RAY start --head --port=${RAY_PORT} --num-cpus=${RAY_NODE_CPUS} --num-gpus=${RAY_NODE_GPUS} --block & \
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
            $RAY start --block --address=$head_node_hostname:${RAY_PORT}  --num-cpus=${RAY_NODE_CPUS} --num-gpus=${RAY_NODE_GPUS}" &
echo "DEBUG: Ray worker srun command issued. Waiting for it to connect..."
sleep 1000 # TO-DO: Exit gracefully

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
