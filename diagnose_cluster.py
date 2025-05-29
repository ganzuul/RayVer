import os
import json
import sys
import time
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

import logging
from rich.logging import RichHandler
from rich.console import Console

import ray

# --- Enhanced Error Handling ---
class InferenceError(Exception):
    """Custom exception for inference errors"""
    pass

def setup_logging():
    """Enhanced logging setup with multiple handlers"""
    log_dir = os.getenv('LOGGING_DIR_ENV', './ray_data_vllm_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    job_id = os.getenv("SLURM_JOB_ID", "local")
    log_file = os.path.join(log_dir, f'ray_data_vllm_{job_id}_{datetime.now().strftime("%H%M%S")}.log')
    
    # Create rich console for stderr
    try:
        console_width = max(os.get_terminal_size().columns, 80)
    except (OSError, AttributeError):
        console_width = 120
        
    console = Console(width=console_width, stderr=True)
    rich_handler = RichHandler(
        console=console, 
        rich_tracebacks=True, 
        show_path=False,
        markup=True, 
        log_time_format="[%H:%M:%S]"
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(message)s",
        handlers=[rich_handler, file_handler],
        force=True
    )
    
    return logging.getLogger("RayVLLM")

# --- Environment Configuration ---
def load_config():
    """Load and validate configuration from environment variables"""
    config = {
        # Paths
        'output_dir': os.getenv('OUTPUT_DIR_ENV', './ray_data_vllm_output'),
        'logging_dir': os.getenv('LOGGING_DIR_ENV', './ray_data_vllm_logs'),
        'prompts_file': os.getenv('PROMPTS_FILE_ENV', 'prompts.json'),
    }
    
    # Create directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['logging_dir'], exist_ok=True)
    
    return config

# --- Ray Cluster Diagnostics ---
def diagnose_ray_cluster(logger):
    """Comprehensive Ray cluster diagnostics"""
    try:
        logger.info("=== Ray Cluster Diagnostics ===")
        
        # Basic Ray info
        logger.info(f"Ray version: {ray.__version__}")
        logger.info(f"Ray initialized: {ray.is_initialized()}")
        
        if ray.is_initialized():
            # Cluster resources
            resources = ray.cluster_resources()
            logger.info(f"Cluster resources: {json.dumps(resources, indent=2)}")
            
            # Available resources
            available = ray.available_resources()
            logger.info(f"Available resources: {json.dumps(available, indent=2)}")
            
            # Node information
            nodes = ray.nodes()
            logger.info(f"Number of nodes: {len(nodes)}")
            for i, node in enumerate(nodes):
                logger.info(f"Node {i}: {node.get('NodeManagerAddress', 'Unknown')} - "
                          f"Alive: {node.get('Alive', False)} - "
                          f"Resources: {node.get('Resources', {})}")
        
        # Test simple Ray task
        @ray.remote
        def test_task():
            import torch
            return {
                'hostname': os.uname().nodename,
                'cuda_available': torch.cuda.is_available() if 'torch' in sys.modules else False,
                'cuda_devices': torch.cuda.device_count() if 'torch' in sys.modules else 0
            }
        
        logger.info("Testing Ray task execution...")
        futures = [test_task.remote() for _ in range(min(2, len(ray.nodes())))]
        results = ray.get(futures, timeout=30)
        
        for i, result in enumerate(results):
            logger.info(f"Task {i} result: {result}")
            
    except Exception as e:
        logger.error(f"Ray diagnostics failed: {e}")
        logger.error(traceback.format_exc())

# --- Main Function ---
def main():
    logger = setup_logging()
    
    try:
        logger.info("=== Starting Enhanced Ray Data vLLM Inference ===")
        
        # Load configuration
        config = load_config()
        logger.info(f"Configuration loaded: {json.dumps({k: v for k, v in config.items() if 'token' not in k.lower()}, indent=2)}")
        
        # Initialize Ray
        ray_address = os.environ.get("RAY_ADDRESS")
        logger.info(f"Ray address: {ray_address or 'Local cluster'}")
        
        if ray_address:
            logger.info("Connecting to existing Ray cluster...")
            ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=True)
        else:
            logger.info("Starting local Ray cluster...")
            ray.init(ignore_reinit_error=True, log_to_driver=True)
        
    except Exception as e:
        logger.error(f"Could not connect to Ray: {e}")
        ray.shutdown()

    # Diagnose cluster
    diagnose_ray_cluster(logger)
 