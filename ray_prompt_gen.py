import os
import json
from datetime import datetime
import logging
import sys
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

from rich.logging import RichHandler
from rich.console import Console

import ray
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

# --- Pydantic Model Definitions (as provided previously) ---
# Best practice: Move these to a separate models.py file and import them
# For this example, they are included directly.

DifficultyLevel = Literal["Beginner", "Intermediate"]
SimulatedSource = Literal["Ask Ubuntu Forum", "Arch Wiki Discussion", "Reddit r/linuxquestions", "Stack Overflow"]
DiagnosticNextStepType = Literal["RunAnotherDiagnosticCommand", "ReadyForSolution", "OutputNeedsUserClarification"]

class ProblemMetadata(BaseModel):
    category: str = Field(..., description="The broad technical area of the problem.", examples=["Package Management", "Networking", "Hardware (Drivers/Peripherals)"])
    subcategory: str = Field(..., description="A more specific description within the category.", examples=["Dependency Conflict", "Wi-Fi Connectivity", "USB Device Not Recognized"])
    difficulty: DifficultyLevel = Field(..., description="Estimation of the technical difficulty level.")
    simulated_source: SimulatedSource = Field(..., description="The type of platform or source this interaction mimics.")
    tags: List[str] = Field(..., description="Relevant keywords or technical terms.", examples=[["#apt", "#usb", "#dmesg"]])
    requires_diagnostic: bool = Field(..., description="True if this problem commonly requires diagnostic commands.")
    tested_in_vm_recommended: bool = Field(..., description="True if the solution carries significant risk and VM testing is advised.")

class SystemContext(BaseModel):
    distribution: str = Field(..., examples=["Ubuntu 22.04 LTS", "Fedora 39","openSUSE Leap 15.0"])
    desktop_environment: Optional[str] = Field(None, examples=["GNOME 42", "KDE Plasma 5.27"])
    kernel_version: str = Field(..., description="Output of uname -r.", examples=["5.15.0-86-generic"])
    hardware_summary: Optional[str] = Field(None, description="Relevant hardware, especially for hardware-related issues.", examples=["XP-Pen Deco 01 V2 tablet, AMD Ryzen 7 5700U"])
    other_relevant_info: Optional[str] = Field(None, description="Any other specific context like dual-boot setup, recent changes.")

class UserProblemPost(BaseModel):
    title: str = Field(..., description="A concise, attention-grabbing summary of the problem.")
    persona_description: str = Field(..., description="User's background, experience level, and comfort with CLI/GUI.", examples=["Intermediate Ubuntu user comfortable with terminal basics but unfamiliar with kernel modules"])
    problem_description: str = Field(..., description="User's narrative explaining the issue.")
    error_messages: Optional[List[str]] = Field(default_factory=list, description="Exact text of any errors or relevant log snippets.", examples=[["dmesg output shows: [ 1234.567890] usb 3-2: device descriptor read/64, error -71"]])
    system_context: SystemContext
    steps_already_tried: Optional[List[str]] = Field(default_factory=list, description="Actions the user has already attempted.", examples=[["Tried different USB ports and cables", "Rebooted multiple times"]])
    user_goal: str = Field(..., description="Clear statement of what the user wants to achieve.")

class DiagnosticStep(BaseModel):
    reason_for_command: str = Field(..., description="Why this diagnostic command is being run.")
    diagnostic_command: str = Field(..., description="The exact non-modifying command.", examples=["dmesg | grep -i usb", "lsusb"])
    simulated_output: str = Field(..., description="Simulated output of the command, using code block formatting.")
    interpretation: str = Field(..., description="Expert's interpretation of the simulated_output.")
    informs_next_step_type: DiagnosticNextStepType = Field(..., description="What the interpretation suggests for the next step.")

class RelevantExpertContext(BaseModel):
    experience_snippet: str = Field(..., examples=["Having resolved numerous USB device conflicts on Ubuntu systems..."])
    common_pitfalls: List[str] = Field(..., description="Common mistakes or dangerous alternatives to avoid.", examples=[["Blacklisting essential USB drivers without understanding dependencies"]])
    effective_strategy_overview: str = Field(..., description="General approach or reasoning behind the solution.")

class StepByStepFixItem(BaseModel):
    step_number: int
    instruction: str = Field(..., description="User-friendly explanation of what to do.")
    commands: Optional[List[str]] = Field(default_factory=list, description="Exact command(s) to run, if applicable.")
    gui_path: Optional[str] = Field(None, description="Path through GUI, if applicable.")
    explanation: str = Field(..., description="What the command or GUI action does.")
    warning: Optional[str] = Field(None, description="Step-specific caution if potentially risky.")

class VerificationStep(BaseModel):
    instruction: str
    commands: Optional[List[str]] = Field(default_factory=list)
    expected_output: Optional[str] = Field(None, description="What output should look like if successful.")

class AlternativeSolution(BaseModel):
    description: str
    commands: Optional[List[str]] = Field(default_factory=list)
    warnings: Optional[List[str]] = Field(default_factory=list)

class ExpertSolutionReply(BaseModel):
    acknowledgement: str
    relevant_expert_context: RelevantExpertContext
    step_by_step_fix: List[StepByStepFixItem]
    explanation: str = Field(..., description="Summary explanation of why the overall solution works and the likely root cause.")
    warnings: List[str] = Field(..., description="Prominent general warnings about the solution or problem type.")
    verification_steps: List[VerificationStep]
    alternative_solutions: Optional[List[AlternativeSolution]] = Field(default_factory=list)
    root_cause_simplified: str

class LinuxAdminDataEntry(BaseModel):
    problem_metadata: ProblemMetadata
    user_problem_post: UserProblemPost
    diagnostic_reasoning_process: List[DiagnosticStep]
    expert_solution_reply: ExpertSolutionReply

# --- End of Pydantic Model Definitions ---

# Generate the JSON schema for the StructuredOutput model
structured_output_schema = LinuxAdminDataEntry.model_json_schema()

# --- Environment Variable Setup ---
OUTPUT_DIR = os.getenv('OUTPUT_DIR_ENV', './ray_data_vllm_output')
LOGGING_DIR = os.getenv('LOGGING_DIR_ENV', './ray_data_vllm_logs')
HF_HOME = os.getenv('HF_HOME') # Keep for potential tokenizer downloads by vLLM
HF_HUB_CACHE = os.getenv('HF_HUB_CACHE') # Keep for potential model downloads by vLLM
TOKEN = os.getenv('HF_TOKEN') # Keep if model needs auth

MODEL_ID_OR_PATH = os.getenv("MODEL_ID_OR_PATH_ENV", "mistralai/Mixtral-8x7B-Instruct-v0.1")
PROMPTS_FILE = os.getenv("PROMPTS_FILE_ENV", "prompts.json") # JSON list of strings
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE_ENV", "1"))
PIPELINE_PARALLEL_SIZE = int(os.getenv("PIPELINE_PARALLEL_SIZE_ENV", "1")) # Used for concurrency
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION_ENV", "0.90"))
MODEL_DTYPE = os.getenv("MODEL_DTYPE_ENV", "bfloat16")
VLLM_DISTRIBUTED_EXECUTOR_BACKEND = os.getenv("VLLM_DISTRIBUTED_EXECUTOR_BACKEND", "ray")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5")) # Allow 0.0 for greedy
TOP_P = float(os.getenv("TOP_P", "1.0")) # Default to 1.0 for Ray Data example if not set
TOP_K = int(os.getenv("TOP_K", "-1")) # Default to -1 (disabled) for Ray Data example if not set

MAX_MODEL_LEN_ENV = os.getenv("MAX_MODEL_LEN_ENV")
MAX_MODEL_LEN_PARAM = None
if MAX_MODEL_LEN_ENV:
    try:
        MAX_MODEL_LEN_PARAM = int(MAX_MODEL_LEN_ENV)
    except ValueError:
        pass # Logger will warn later

RAY_DATA_BATCH_SIZE = int(os.getenv("RAY_DATA_BATCH_SIZE", "16"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)

# --- Logger Setup ---
log_file_path = os.path.join(LOGGING_DIR, f'ray_data_vllm_inference_{os.getenv("SLURM_JOB_ID", "local")}.log')
try:
    console_width = int(os.get_terminal_size().columns)
except OSError:
    console_width = int(os.getenv("COLUMNS", "120"))
console_width = max(console_width, 80)
rich_console_stderr = Console(width=console_width, stderr=True)
rich_handler = RichHandler(
    console=rich_console_stderr, rich_tracebacks=True, show_path=False,
    markup=True, log_time_format="[%H:%M:%S]"
)
logging.basicConfig(
    level=logging.INFO, format="%(name)s: %(message)s", datefmt="[%X]",
    handlers=[rich_handler, logging.FileHandler(log_file_path)],
    force=True
)
logger = logging.getLogger("RayData_vLLM")

if MAX_MODEL_LEN_ENV and MAX_MODEL_LEN_PARAM is None:
    logger.warning(f"MAX_MODEL_LEN_ENV ('{MAX_MODEL_LEN_ENV}') is set but not a valid integer. Letting vLLM infer max_model_len.")

# --- Preprocessing Function ---
def preprocess_prompts(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares a row (expected to contain a 'prompt' key) for the LLM.
    """
    # Ensure temperature is not strictly 0 for sampling if top_p or top_k active,
    # or handle as per vLLM's specific guidance for greedy + guided decoding.
    # For guided_json, temperature can often be low or 0.
    current_temperature = TEMPERATURE
    current_top_p = TOP_P
    current_top_k = TOP_K

    # vLLM typically handles temperature=0 as greedy.
    # If temperature is 0, top_k and top_p are usually ignored or should be set to disable them.
    if current_temperature == 0.0:
        current_top_p = 1.0 # vLLM default for greedy
        current_top_k = -1  # vLLM default for greedy

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are an AI assistant specialized in generating high-quality synthetic data. Generate structured JSON output that conforms to the provided schema. Be creative, diverse, and ensure the output is meaningful and realistic."
            },
            {
                "role": "user",
                "content": row["prompt"]
            },
        ],
        "sampling_params": dict(
            temperature=current_temperature,
            max_tokens=MAX_NEW_TOKENS,
            top_p=current_top_p,
            top_k=current_top_k,
            detokenize=False, # Recommended by Ray Data example for vLLM
            guided_decoding=dict(json=structured_output_schema)
        ),
    }

# --- Postprocessing Function ---
def postprocess_responses(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes the LLM's output row.
    The input `row` will contain the original columns from `preprocess_prompts`'s input
    plus a "generated_text" column from the LLM.
    """
    return {"generated_json_string": row["generated_text"]}


def main():
    logger.info("--- Starting Ray Data vLLM Structured Inference Script ---")
    logger.info(f"SLURM_JOB_ID: {os.getenv('SLURM_JOB_ID', 'N/A')}")
    ray_address = os.environ.get("RAY_ADDRESS")
    logger.info(f"RAY_ADDRESS: {ray_address if ray_address else 'Not Set - Ray will attempt to start a new cluster or connect locally'}")
    
    try:
        logger.info("Initializing Ray...")
        ray.init(address=ray_address, ignore_reinit_error=True)
        logger.info("Ray initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info(f"Logging Directory: {LOGGING_DIR}")
    logger.info(f"Model ID / Path: {MODEL_ID_OR_PATH}")
    logger.info(f"Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
    logger.info(f"Pipeline Parallel Size (Concurrency): {PIPELINE_PARALLEL_SIZE}")
    logger.info(f"GPU Memory Utilization: {GPU_MEMORY_UTILIZATION}")
    logger.info(f"Model Data Type: {MODEL_DTYPE}")
    if MAX_MODEL_LEN_PARAM:
        logger.info(f"Max Model Length (passed to vLLM): {MAX_MODEL_LEN_PARAM}")
    else:
        logger.info(f"Max Model Length: Will be inferred by vLLM from model config.")
    logger.info(f"Prompts File: {PROMPTS_FILE}")
    # logger.info(f"Ray Data Batch Size: {RAY_DATA_BATCH_SIZE}")
    logger.info(f"Generation Params (used for vLLM SamplingParams) - Max New Tokens: {MAX_NEW_TOKENS}, Temp: {TEMPERATURE}, Top P: {TOP_P}, Top K: {TOP_K}")
    logger.info(f"Schema for guided decoding: {json.dumps(structured_output_schema, indent=2)}")

    # --- Configure vLLMEngineProcessorConfig ---
    # vllm/vllm/engine/arg_utils.py
    logger.info("Configuring vLLMEngineProcessorConfig...")
    engine_kwargs_config = {
        #"guided_decoding_backend": "xgrammar",
        "dtype": MODEL_DTYPE,
        #"download_dir": SHARED_MODEL_CACHE,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "distributed_executor_backend": VLLM_DISTRIBUTED_EXECUTOR_BACKEND,
        "trust_remote_code": True,
        "enforce_eager": True, # Recommended for stability with Ray Data
        # "enable_expert_parallel": True,
    }
    if MAX_MODEL_LEN_PARAM:
        engine_kwargs_config["max_model_len"] = MAX_MODEL_LEN_PARAM
    
    processor_config = vLLMEngineProcessorConfig(
        model_source=MODEL_ID_OR_PATH,
        engine_kwargs=engine_kwargs_config,
        #batch_size=RAY_DATA_BATCH_SIZE,
    )
    logger.info("vLLMEngineProcessorConfig configured.")

    # --- Build LLM Processor ---
    logger.info("Building LLM processor...")
    try:
        llm_processor = build_llm_processor(
            processor_config,
            preprocess=preprocess_prompts,
            postprocess=postprocess_responses
        )
        logger.info("LLM processor built successfully.")
    except Exception as e:
        logger.error(f"Failed to build LLM processor: {e}", exc_info=True)
        if ray.is_initialized():
            ray.shutdown()
        sys.exit(1)

    # --- Load Input Prompts ---
    logger.info(f"Loading prompts from {PROMPTS_FILE}")
    try:
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            prompts_to_process_list = json.load(f)
        if not isinstance(prompts_to_process_list, list) or not all(isinstance(p, str) for p in prompts_to_process_list):
            logger.error(f"Prompts file {PROMPTS_FILE} should contain a JSON list of strings.")
            if ray.is_initialized():
                ray.shutdown()
            sys.exit(1)
        logger.info(f"Loaded {len(prompts_to_process_list)} prompts.")
    except FileNotFoundError:
        logger.error(f"Prompts file not found: {PROMPTS_FILE}")
        if ray.is_initialized():
            ray.shutdown()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load/parse prompts from {PROMPTS_FILE}: {e}", exc_info=True)
        if ray.is_initialized():
            ray.shutdown()
        sys.exit(1)

    if not prompts_to_process_list:
        logger.info("No prompts to process. Exiting.")
        if ray.is_initialized():
            ray.shutdown()
        return

    # Convert to Ray Dataset
    input_dataset = ray.data.from_items([{"prompt": p} for p in prompts_to_process_list])
    logger.info(f"Created Ray Dataset with {input_dataset.count()} items.")

    # --- Apply Processor and Materialize ---
    logger.info("Applying LLM processor to the dataset...")
    try:
        output_dataset = llm_processor(input_dataset)
        logger.info("LLM processor applied. Materializing output...")
        materialized_output = output_dataset.materialize()
        logger.info("Output materialized successfully.")
    except Exception as e:
        logger.error(f"Error during LLM processing or materialization: {e}", exc_info=True)
        if ray.is_initialized():
            ray.shutdown()
        sys.exit(1)
        
    # --- Save Outputs ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.getenv("SLURM_JOB_ID", "localrun")
    output_file_name = f"output_ray_data_vllm_structured_{job_id}_{timestamp}.jsonl"
    output_file_path = os.path.join(OUTPUT_DIR, output_file_name)

    logger.info(f"Saving results to {output_file_path}")
    saved_count = 0
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            for item in materialized_output.iter_rows():
                # Item is expected to be {"original_prompt": "...", "generated_json_string": "..."}
                # We want to save the generated_json_string which should be a JSON representation
                # of our StructuredOutput model.
                # For consistency with previous structure, let's ensure we are saving a dict that
                # includes the original prompt and the parsed generated_data.
                
                output_entry = {
                    "input_prompt": item.get("original_prompt"),
                    # The "generated_json_string" IS the JSON string for StructuredOutput's generated_data part,
                    # assuming the system prompt + schema guided it correctly.
                    # To fit the StructuredOutput model, we'd parse generated_json_string
                    # and place its content into 'generated_data'.
                    # However, the task is to save the item["generated_json_string"].
                    # Let's save the direct output from postprocess for now.
                    "generated_output_json_string": item.get("generated_json_string"),
                    # "generation_details": "Generated by Ray Data vLLM" # Optional: add static details
                }
                f.write(json.dumps(output_entry) + "\n")
                saved_count += 1
        logger.info(f"Successfully saved {saved_count} results.")
    except Exception as e:
        logger.error(f"Failed to save results: {e}", exc_info=True)


    if ray.is_initialized():
        logger.info("Shutting down Ray.")
        ray.shutdown()
    
    logger.info("--- Ray Data vLLM Structured Inference Script Finished ---")
    print("RAY DATA VLLM STRUCTURED INFERENCE SCRIPT COMPLETED SUCCESSFULLY.", flush=True)

if __name__ == "__main__":
    main()
