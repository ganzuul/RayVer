#!/bin/bash
#SBATCH --job-name=ray-diag-minimal
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

# --- Standard LUMI Modules & AI Bindings ---
#module purge
module use /appl/local/csc/modulefiles/
module load pytorch/2.5
RAY="python3 -m ray.scripts.scripts"

# --- NCCL, HIP, MIOpen Settings ---
#export NCCL_NET=Socket # fallback
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB # or 3
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# --- Ray Cluster Setup (2 nodes) ---
nodes_str=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
mapfile -t nodes_array <<< "$nodes_str"
head_node_hostname=${nodes_array[0]}
worker_node_hostname=${nodes_array[1]}
RAY_PORT=${RAY_PORT:-6379}

# Get the IP address of the head node
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

# Commands to run inside Singularity for starting Ray nodes
RAY_NODE_SETUP_CMDS="export RAY_DISABLE_MEMORY_MONITOR=1; \
ulimit -n 65536; "
# Start Ray Head
echo "DEBUG: Starting Ray head node on $head_node_hostname"
srun --nodes=1 --ntasks=1 -w "$head_node_hostname" --jobid $SLURM_JOBID --cpus-per-task="$RAY_NODE_CPUS" --gpus="$RAY_NODE_GPUS" \
    bash -c "${RAY_NODE_SETUP_CMDS} \
            echo 'Which Ray: '; which ray; \
            $RAY start --head --port=${RAY_PORT} --num-cpus=${RAY_NODE_CPUS} --num-gpus=${RAY_NODE_GPUS} --block & \
            sleep 30;
            echo 'Python client environment:'; \
            env | grep -E 'RAY_|SLURM_|NCCL_|HSA_'; \
            $RAY status
            python3 ray_diag_test.py" &
RAY_HEAD_PID=$!
echo "DEBUG: Ray head process started with PID $RAY_HEAD_PID. Waiting for initialization..."
sleep 5

# Start Ray Worker
echo "DEBUG: Starting Ray worker on $worker_node_hostname, connecting to $head_node_hostname:${RAY_PORT}"
srun --nodes=1 --ntasks=1 --exclude=$head_node_hostname --jobid $SLURM_JOBID --cpus-per-task="$RAY_NODE_CPUS" --gpus="$RAY_NODE_GPUS" \
    bash -c "${RAY_NODE_SETUP_CMDS} \
            $RAY start --block --address=$head_node_hostname:${RAY_PORT}  --num-cpus=${RAY_NODE_CPUS} --num-gpus=${RAY_NODE_GPUS}" &
echo "DEBUG: Ray worker srun command issued. Waiting for it to connect..."
sleep 60

# --- Log Collection ---
# Not happening because of head and worker ntasks still taking up the available slots
# Try Shutting down Ray before collecting logs
echo "DEBUG: Collecting Ray logs..."

# Head Node Log Collection
head_log_dir="$OUTPUT_DIR/ray_head_logs_${head_node_hostname}_$SLURM_JOB_ID"
echo "DEBUG: Collecting Ray head logs from $head_node_hostname into $head_log_dir"
srun --nodes=1 --ntasks=1 -w "$head_node_hostname" --jobid $SLURM_JOBID \
    bash -c "mkdir -p \"$head_log_dir\" && cp -rv /tmp/ray/session_latest/logs/* \"$head_log_dir/\" || echo 'No Ray head logs found or cp failed on $head_node_hostname.'"

# Worker Node Log Collection
worker_log_dir="$OUTPUT_DIR/ray_worker_logs_${worker_node_hostname}_$SLURM_JOB_ID"
echo "DEBUG: Collecting Ray worker logs from $worker_node_hostname into $worker_log_dir"
srun --nodes=1 --ntasks=1 -w "$worker_node_hostname" --jobid $SLURM_JOBID \
    bash -c "mkdir -p \"$worker_log_dir\" && cp -rv /tmp/ray/session_latest/logs/* \"$worker_log_dir/\" || echo 'No Ray worker logs found or cp failed on $worker_node_hostname.'"

echo "DEBUG: Ray log collection finished."

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

# ntasks depleted, won't be tasked
echo "Attempting to stop Ray cluster..."
srun --nodes=1 --ntasks=1 -w "$head_node_hostname" --jobid $SLURM_JOBID \
    bash -c "${RAY_NODE_SETUP_CMDS}$RAY stop" &

echo "Script finished."
