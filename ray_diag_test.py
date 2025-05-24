import ray
import os
import sys
import traceback

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# RAY_ADDRESS should be set by the batch script e.g. ${head_node_ip}:${RAY_PORT}
ray_address = os.environ.get("RAY_ADDRESS")
print(f"Attempting to connect to Ray at: {ray_address}")

if not ray_address:
    print("ERROR: RAY_ADDRESS environment variable not set. Cannot initialize Ray.")
    sys.exit(1)

try:
    print("Initializing Ray...")
    # For this diagnostic, we expect RAY_ADDRESS to be explicitly set.
    ray.init(address=ray_address, ignore_reinit_error=True)
    print("Successfully connected to Ray!")
    
    print("\nRay Cluster Nodes:")
    nodes = ray.nodes()
    print(nodes)
    if len(nodes) < 2:
        print(f"WARNING: Expected at least 2 nodes (1 head, 1 worker), but found {len(nodes)}.")
    else:
        print(f"Found {len(nodes)} nodes, which is expected for a 2-node setup (1 head, 1 worker).")

    print("\nCluster Resources:")
    print(ray.cluster_resources())
    
    print("\nAvailable Resources:")
    print(ray.available_resources())

    print("\nRelevant Environment Variables (seen by Python script):")
    relevant_vars = [
        "SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_NNODES",
        "SLURM_NODEID", "SLURMD_NODENAME", "SLURM_PROCID", "SLURM_SUBMIT_HOST",
        "SLURM_CPUS_ON_NODE", "SLURM_GPUS_ON_NODE",
        "RAY_ADDRESS",
        "PYTHONPATH", "PATH", "LD_LIBRARY_PATH",
        "SINGULARITY_NAME", "SINGULARITY_CONTAINER",
        "NCCL_SOCKET_IFNAME", "HSA_FORCE_FINE_GRAIN_PCIE",
        "MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR"
    ]
    for var in relevant_vars:
        print(f"{var}: {os.environ.get(var, 'Not Set')}")

except Exception as e:
    print(f"ERROR: Error connecting to Ray or during Ray operations: {e}")
    traceback.print_exc()
    sys.exit(1) # Exit with error if Ray operations fail
finally:
    if ray.is_initialized():
        print("\nShutting down Ray connection.")
        ray.shutdown()
    print("Python script finished.")
