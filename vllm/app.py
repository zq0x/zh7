from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import torch
import pynvml
from transformers import AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NVML once (e.g., at application startup)
def initialize_nvml():
    try:
        pynvml.nvmlInit()
        logger.info('[START] [initialize_nvml] NVML initialized successfully.')
    except Exception as e:
        logger.error(f'[START] [initialize_nvml] Failed to initialize NVML: {e}')

initialize_nvml()

# Shutdown NVML once (e.g., at application exit)
def shutdown_nvml():
    try:
        pynvml.nvmlShutdown()
        logger.info('[START] [shutdown_nvml] NVML shutdown successfully.')
    except Exception as e:
        logger.error(f'[START] [shutdown_nvml] Failed to shutdown NVML: {e}')

# Check if CUDA is available
def cuda_support_bool():
    try:
        res_cuda_support = torch.cuda.is_available()
        if res_cuda_support:
            logger.info('[START] [cuda_support_bool] CUDA found')
        else:
            logger.error('[START] [cuda_support_bool] CUDA not supported!')
        return res_cuda_support
    except Exception as e:
        logger.error(f'[START] [cuda_support_bool] {e}')
        return False

# Return the number of CUDA GPUs available
def cuda_device_count():
    try:
        res_gpu_int = torch.cuda.device_count()
        res_gpu_int_arr = [i for i in range(res_gpu_int)]
        logger.info(f'[START] [cuda_support_bool] {str(res_gpu_int)}x GPUs found {res_gpu_int_arr}')
        return res_gpu_int_arr
    except Exception as e:
        logger.error(f'[START] [cuda_device_count] Failed to get device_count. Using default [0]. Error: {e}')
        return [0]

gpu_int_arr = cuda_device_count()

# FastAPI app
app = FastAPI()

# Global LLM instance
llm_instance = None

@app.get("/")
async def root():
    return f'Hello from server :DDDDDDDDDDDDDDDDDDDDD {os.getenv("CONTAINER_PORT")}!'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("CONTAINER_PORT")))
    





# if req_data["req_model"] not in llm_instances:
#     print(f'MODEL NOT IN STANCES EGAAAAAAAL')
# llm_instances[req_data["req_model"]] = LLM(model=req_data["req_model"], task=req_data["req_task"])

# print("llm_instances")
# print(llm_instances)
# sampling_params = SamplingParams(int(temperature=req_data["req_temperature"]), top_p=0.9, max_tokens=100)

# outputs = llm_instances[req_data["req_model"]].generate([req_data["req_prompt"]], sampling_params)
# return JSONResponse({"result_status": 200, "result_data": outputs[0].outputs[0].text})
