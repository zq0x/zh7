import os
import requests
import json
import subprocess
import sys
import ast
import time
from datetime import datetime
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import huggingface_hub
from huggingface_hub import snapshot_download
import redis
import gradio as gr
import logging
import psutil

docker_container_list = []
current_models_data = []
db_gpu_data = []
db_gpu_data_len = ''

try:
    r = redis.Redis(host="redis", port=6379, db=0)
    db_gpu = json.loads(r.get('db_gpu'))
    # print(f'db_gpu: {db_gpu} {len(db_gpu)}')
    db_gpu_data_len = len(db_gpu_data)
except Exception as e:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')

LOGFILE_CONTAINER = './logs/logfile_container_frontend.log'
os.makedirs(os.path.dirname(LOGFILE_CONTAINER), exist_ok=True)
logging.basicConfig(filename=LOGFILE_CONTAINER, level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def load_log_file(req_container_id):
    print(f' **************** GOT LOG FILE REQUEST FOR CONTAINER ID: {req_container_id}')
    logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] **************** GOT LOG FILE REQUEST FOR CONTAINER ID: {req_container_id}')
    try:
        with open(LOGFILE_CONTAINER, 'r') as file:
            lines = file.readlines()
            return ''.join(lines)
    except Exception as e:
        return f'{e}'


def get_container_data():
    try:
        res_container_data_all = json.loads(r.get('db_container'))
        return res_container_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
   
def get_network_data():
    try:
        res_network_data_all = json.loads(r.get('db_network'))
        return res_network_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e

def get_gpu_data():
    try:
        res_gpu_data_all = json.loads(r.get('db_gpu'))
        return res_gpu_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e

def get_docker_container_list():
    global docker_container_list
    response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"list"})
    # print(f'[get_docker_container_list] response: {response}')
    res_json = response.json()
    # print(f'[get_docker_container_list] res_json: {res_json}')
    docker_container_list = res_json.copy()
    if response.status_code == 200:
        # print(f'[get_docker_container_list] res = 200')
        return res_json
    else:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        # logging.info(f'[get_docker_container_list] [get_docker_container_list] res_json: {res_json}')
        return f'Error: {response.status_code}'

def docker_api_logs(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"logs","req_model":req_model})
        res_json = response.json()
        return f'{res_json["result_data"]}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action stop'

def docker_api_network(req_container_name):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"network","req_container_name":req_container_name})
        res_json = response.json()
        if res_json["result"] == 200:
            return f'{res_json["result_data"]["networks"]["eth0"]["rx_bytes"]}'
        else:
            return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action network {res_json}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action network {e}'
    
def docker_api_start(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"start","req_model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action start {e}'

def docker_api_stop(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"stop","req_model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action stop {e}'

def docker_api_delete(req_model):
    try:
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"delete","req_model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action delete {e}'

def docker_api_create(req_model, req_pipeline_tag, req_port_model, req_port_vllm):
    try:
        req_container_name = str(req_model).replace('/', '_')
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method":"create","req_container_name":req_container_name,"req_model":req_model,"req_runtime":"nvidia","req_port_model":req_port_model,"req_port_vllm":req_port_vllm})
        response_json = response.json()
        
        new_entry = [{
            "gpu": 8,
            "path": f'/home/cloud/.cache/huggingface/{req_model}',
            "container": "0",
            "container_status": "0",
            "running_model": req_container_name,
            "model": req_model,
            "pipeline_tag": req_pipeline_tag,
            "port_model": req_port_model,
            "port_vllm": req_port_vllm
        }]
        r.set("db_gpu", json.dumps(new_entry))

        print(response_json["result"])
        if response_json["result"] == 200:
            return f'{response_json["result_data"]}'
        else:
            return f'Create result ERR no container_id: {str(response_json)}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'error docker_api_create'

def search_models(query):
    try:
        global current_models_data    
        response = requests.get(f'https://huggingface.co/api/models?search={query}')
        response_models = response.json()
        current_models_data = response_models.copy()
        model_ids = [m["id"] for m in response_models]
        return gr.update(choices=model_ids, value=response_models[0]["id"], show_label=True, label=f'found {len(response_models)} models!')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')

def calculate_model_size(json_info): # to fix    
    try:
        d_model = json_info.get("hidden_size") or json_info.get("d_model")
        num_hidden_layers = json_info.get("num_hidden_layers", 0)
        num_attention_heads = json_info.get("num_attention_heads") or json_info.get("decoder_attention_heads") or json_info.get("encoder_attention_heads", 0)
        intermediate_size = json_info.get("intermediate_size") or json_info.get("encoder_ffn_dim") or json_info.get("decoder_ffn_dim", 0)
        vocab_size = json_info.get("vocab_size", 0)
        num_channels = json_info.get("num_channels", 3)
        patch_size = json_info.get("patch_size", 16)
        torch_dtype = json_info.get("torch_dtype", "float32")
        bytes_per_param = 2 if torch_dtype == "float16" else 4
        total_size_in_bytes = 0
        
        if json_info.get("model_type") == "vit":
            embedding_size = num_channels * patch_size * patch_size * d_model
            total_size_in_bytes += embedding_size

        if vocab_size and d_model:
            embedding_size = vocab_size * d_model
            total_size_in_bytes += embedding_size

        if num_attention_heads and d_model and intermediate_size:
            attention_weights_size = num_hidden_layers * (d_model * d_model * 3)
            ffn_weights_size = num_hidden_layers * (d_model * intermediate_size + intermediate_size * d_model)
            layer_norm_weights_size = num_hidden_layers * (2 * d_model)

            total_size_in_bytes += (attention_weights_size + ffn_weights_size + layer_norm_weights_size)

        if json_info.get("is_encoder_decoder"):
            encoder_size = num_hidden_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            decoder_layers = json_info.get("decoder_layers", 0)
            decoder_size = decoder_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            
            total_size_in_bytes += (encoder_size + decoder_size)

        return total_size_in_bytes
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return 0

def get_info(selected_id):    
    global current_models_data    
    res_model_data = {
        "search_data" : "",
        "model_id" : selected_id,
        "pipeline_tag" : "",
        "transformers" : "",
        "private" : "",
        "downloads" : ""
    }

    try:
        for item in current_models_data:
            if item['id'] == selected_id:
                
                res_model_data["search_data"] = item
                
                if "pipeline_tag" in item:
                    res_model_data["pipeline_tag"] = item["pipeline_tag"]
  
                if "tags" in item:
                    if "transformers" in item["tags"]:
                        res_model_data["transformers"] = True
                    else:
                        res_model_data["transformers"] = False
                                    
                if "private" in item:
                    res_model_data["private"] = item["private"]
                                  
                if "downloads" in item:
                    res_model_data["downloads"] = item["downloads"]
                  
                container_name = str(res_model_data["model_id"]).replace('/', '_')
                
                return res_model_data["search_data"], res_model_data["model_id"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
                
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"]

def get_additional_info(selected_id):    
        res_model_data = {
            "hf_data" : "",
            "config_data" : "",
            "model_id" : selected_id,
            "size" : 0,
            "gated" : ""
        }                
        try:
            try:
                model_info = huggingface_hub.model_info(selected_id)
                model_info_json = vars(model_info)
                res_model_data["hf_data"] = model_info_json
                
                if "gated" in model_info.__dict__:
                    res_model_data['gated'] = model_info_json["gated"]
                
                if "safetensors" in model_info.__dict__:                        
                    safetensors_json = vars(model_info.safetensors)
                    res_model_data['size'] = safetensors_json["total"]
            
            except Exception as get_model_info_err:
                res_model_data['hf_data'] = f'{get_model_info_err}'
                pass
                    
            try:
                url = f'https://huggingface.co/{selected_id}/resolve/main/config.json'
                response = requests.get(url)
                if response.status_code == 200:
                    response_json = response.json()
                    res_model_data["config_data"] = response_json                     
                else:
                    res_model_data["config_data"] = f'{response.status_code}'
                    
            except Exception as get_config_json_err:
                res_model_data["config_data"] = f'{get_config_json_err}'
                pass                       
            
            if res_model_data["size"] == 0:
                try:
                    res_model_data["size"] = calculate_model_size(res_model_data["config_data"]) 
                except Exception as get_config_json_err:
                    res_model_data["size"] = 0
    
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["model_id"], res_model_data["size"], res_model_data["gated"]
    
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["model_id"], res_model_data["size"], res_model_data["gated"]

def gr_load_check(selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_gated):
    if selected_model_pipeline_tag != '' and selected_model_transformers == 'True' and selected_model_private == 'False' and selected_model_gated == 'False':
        return gr.update(visible=False), gr.update(value=f'Download {selected_model_id[:12]}...', visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)


def network_to_pd():       
    rows = []
    try:
        network_list = get_network_data()
        # logging.info(f'[network_to_pd] network_list: {network_list}')  # Use logging.info instead of logging.exception
        for entry in network_list:
            # entry_info = ast.literal_eval(entry['info'])  # Parse the string into a dictionary
            # print("entry_info")
            # print(entry_info)
            # entry_info_networks = entry_info.get("networks", {})
            # print("entry_info_networks")
            # print(entry_info_networks)            
            # first_network_key = next(iter(entry_info_networks))
            # print("first_network_key")
            # print(first_network_key)                 
            # first_network_data = entry_info_networks[first_network_key]
            # print("first_network_data")
            # print(first_network_data)                          
            # first_network_data_rx_bytes = first_network_data["rx_bytes"]
            # print("first_network_data_rx_bytes")
            # print(first_network_data_rx_bytes)              
            # print(f"First network data: {first_network_data}")
            
            rows.append({
                "container": entry["container"],
                "current_dl": entry["current_dl"],
                "timestamp": entry["timestamp"],
                "info": entry["info"]
            })
            
            
        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        rows.append({
                "container": "0",
                "current_dl": f'0',
                "timestamp": f'0',
                "info": f'0'
        })
        df = pd.DataFrame(rows)
        return df


def container_to_pd():       
    rows = []
    try:
        container_list = get_container_data()
        # print("container_list")
        # print(container_list)
        # logging.info(f'[container_to_pd] container_list: {container_list}')  # Use logging.info instead of logging.exception
        for entry in container_list:
            container_info = ast.literal_eval(entry['container_info'])  # Parse the string into a dictionary
            rows.append({
                "container_i": entry["container_i"]
            })
        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        rows.append({
                "network_i": "0"
        })
        df = pd.DataFrame(rows)
        return df





def gpu_to_pd():       
    rows = []
    try:
        gpu_list = get_gpu_data()
        # logging.info(f'[gpu_to_pd] gpu_list: {gpu_list}')  # Use logging.info instead of logging.exception
        for entry in gpu_list:
            gpu_info = ast.literal_eval(entry['gpu_info'])  # Parse the string into a dictionary
            rows.append({                
                "current_uuid": gpu_info.get("current_uuid", "0"),
                "gpu_i": entry.get("gpu_i", "0"),
                "gpu_util": gpu_info.get("gpu_util", "0"),
                "mem_util": gpu_info.get("mem_util", "0"),
                "temperature": gpu_info.get("temperature", "0"),
                "fan_speed": gpu_info.get("fan_speed", "0"),
                "power_usage": gpu_info.get("power_usage", "0"),
                "clock_info_graphics": gpu_info.get("clock_info_graphics", "0"),
                "clock_info_mem": gpu_info.get("clock_info_mem", "0"),
                "timestamp": entry.get("timestamp", "0"),
                "cuda_cores": gpu_info.get("cuda_cores", "0"),
                "compute_capability": gpu_info.get("compute_capability", "0"),
                "supported": gpu_info.get("supported", "0"),
                "not_supported": gpu_info.get("not_supported", "0"),
                "status": "ok"
            })

        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')





gpu_to_pd()


def download_from_hf_hub(selected_model_id):
    try:
        selected_model_id_arr = str(selected_model_id).split('/')
        print(f'selected_model_id_arr {selected_model_id_arr}...')       
        model_path = snapshot_download(
            repo_id=selected_model_id,
            local_dir=f'/models/{selected_model_id_arr[0]}/{selected_model_id_arr[1]}'
        )
        return f'Saved to {model_path}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'download error: {e}'



rx_change_arr = []
def check_rx_change():
    try:
        global rx_change_arr
        
        if len(rx_change_arr) > 4:
            last_value = rx_change_arr[-1]
            same_value_count = 0
            for i in range(1,len(rx_change_arr)):
                if rx_change_arr[i*-1] == last_value:
                    same_value_count += 1
                    if same_value_count > 10:
                        return f'Count > 10 Download finished'
                else:
                    return f'Count: {same_value_count} {str(rx_change_arr)}'
            return f'Count: {same_value_count} {str(rx_change_arr)}'   
        else:
            return f'waiting for download ...'     
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'Error rx_change_arr {str(e)}'


prev_bytes_recv = 0
def get_download_speed():
    try:
        global prev_bytes_recv
        global rx_change_arr
        
        print(f'trying to get download speed ...')
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - prev_bytes_recv
        prev_bytes_recv = bytes_recv
        download_speed_kb = download_speed / 1024
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2)      
        bytes_received_mb = bytes_recv
        rx_change_arr.append(rx_change_arr)
        return f'{download_speed_mbit_s:.2f} MBit/s (total: {bytes_received_mb})'
        # return f'{download_speed_kb:.2f} KB/s (total: {bytes_received_mb:.2f})'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'download error: {e}'


VLLM_URL = f'http://container_vllm:{os.getenv("VLLM_PORT")}/vllmt'


# def vllm_api(request: gr.Request): 
# def vllm_api(req_type,max_tokens=None,temperature=None,prompt_in=None):


def vllm_api(
                req_type,
                model=None,
                pipeline_tag=None,
                max_model_len=None,
                enforce_eager=None,
                enable_prefix_caching=None,
                pipeline_parallel_size=None,
                tensor_parallel_size=None,
                max_parallel_loading_workers=None,
                kv_cache_dtype=None,
                port=None,
                swap_space=None,                
                gpu_memory_utilization=None,
                enable_chunked_prefill=None,
                trust_remote_code=None,
                load_format=None,
                dtype=None,
                quantization_param_path=None,
                block_size=None,
                num_lookahead_slots=None,
                seed=None,
                num_gpu_blocks_override=None,
                max_num_batched_tokens=None,
                max_num_seqs=None,
                max_logprobs=None,
                quantization=None,
                max_content_len_to_capture=None,
                tokenizer_pool_size=None,
                tokenizer_pool_type=None,
                tokenizer_pool_extra_config=None,
                enable_lora=None,
                max_loras=None,
                max_lora_rank=None,
                lora_extra_vocab_size=None,
                lora_dtype=None,
                max_cpu_loras=None,
                device=None,
                image_input_type=None,
                image_token_id=None,
                image_input_shape=None,
                image_feature_size=None,
                scheduler_delay_factor=None,
                speculative_model=None,
                num_speculative_tokens=None,
                speculative_max_model_len=None,
                model_loader_extra_config=None,
                engine_use_ray=None,
                disable_log_requests=None,
                max_log_len=None,
                top_p=None,
                temperature=None,
                max_tokens=None,
                prompt_in=None
             ):
    
    try:

        FALLBACK_VLLM_API = {}
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] req_type: {req_type}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] model: {model}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] pipeline_tag: {pipeline_tag}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] max_model_len: {max_model_len}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] tensor_parallel_size: {tensor_parallel_size}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] gpu_memory_utilization: {gpu_memory_utilization}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] top_p: {top_p}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] temperature: {temperature}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] max_tokens: {max_tokens}')
        logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] prompt_in: {prompt_in}')

        if req_type == "load":
            response = "if you see this it didnt work :/"  
            
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] [{req_type}] gpu_memory_utilization: {gpu_memory_utilization}')
            response = requests.post(VLLM_URL, json={
                "req_type":"load",
                "max_model_len":int(max_model_len),
                "tensor_parallel_size":int(tensor_parallel_size),
                "gpu_memory_utilization":float(gpu_memory_utilization),
                "model":str(model)
            })
            if response.status_code == 200:
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] [{req_type}] status_code: {response.status_code}') 
                response_json = response.json()
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] [{req_type}] response_json: {response_json}') 
                response_json["result_data"] = response_json["result_data"]
                return response_json["result_data"]                
            else:
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] [{req_type}] response: {response}')
                FALLBACK_VLLM_API["result_data"] = f'{response}'
                return FALLBACK_VLLM_API
    
    

        if req_type == "generate":
            response = "if you see this it didnt work :/"  
            logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] [{req_type}] temperature: {temperature}')
            response = requests.post(VLLM_URL, json={
                "req_type":"generate",
                "prompt":str(prompt_in),
                "temperature":float(temperature),
                "top_p":float(top_p),
                "max_tokens":int(max_tokens)
            })
            if response.status_code == 200:
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] [{req_type}] status_code: {response.status_code}') 
                response_json = response.json()
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] [{req_type}] response_json: {response_json}') 
                return response_json["result_data"]                
            else:
                logging.info(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] response: {response}')
                FALLBACK_VLLM_API["result_data"] = f'{response}'
                return FALLBACK_VLLM_API
    
    
    except Exception as e:
        logging.exception(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] [vllm_api] Exception occured: {e}', exc_info=True)
        FALLBACK_VLLM_API["result_data"] = f'{e}'
        return FALLBACK_VLLM_API





BACKEND_URL = f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest'

def docker_api(req_type,req_model=None,req_task=None,req_prompt=None,req_temperature=None, req_config=None):
    
    try:
        print(f'got model_config: {req_config} ')
        response = requests.post(BACKEND_URL, json={
            "req_type":req_type,
            "req_model":req_model,
            "req_task":req_task,
            "req_prompt":req_prompt,
            "req_temperature":req_temperature,
            "req_model_config":req_config
        })
        
        if response.status_code == 200:
            response_json = response.json()
            if response_json["result_status"] != 200:
                logging.exception(f'[docker_api] Response Error: {response_json["result_data"]}')
            return response_json["result_data"]                
        else:
            logging.exception(f'[docker_api] Request Error: {response}')
            return f'Request Error: {response}'
    
    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e




def refresh_container():
    try:
        global docker_container_list
        response = requests.post(f'http://container_backend:{os.getenv("BACKEND_PORT")}/dockerrest', json={"req_method": "list"})
        docker_container_list = response.json()
        return docker_container_list
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'err {str(e)}'






with gr.Blocks() as app:
    gr.Markdown(
        """
        # Welcome!
        Select a Hugging Face model and deploy it on a port
        
        **Note**: _[vLLM supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html)_        
        """)

    inp = gr.Textbox(placeholder="Type in a Hugging Face model or tag", show_label=False, autofocus=True)
    btn = gr.Button("Search")
    out = gr.Textbox(visible=False)

    model_dropdown = gr.Dropdown(choices=[''], interactive=True, show_label=False, visible=False)

    with gr.Row():
        selected_model_id = gr.Textbox(label="id",visible=False)
        selected_model_container_name = gr.Textbox(label="container_name",visible=False)
        
    with gr.Row():       
        selected_model_pipeline_tag = gr.Textbox(label="pipeline_tag", visible=False)
        selected_model_transformers = gr.Textbox(label="transformers", visible=False)
        selected_model_private = gr.Textbox(label="private", visible=False)
        
    with gr.Row():
        selected_model_size = gr.Textbox(label="size", visible=False)
        selected_model_gated = gr.Textbox(label="gated", visible=False)
        selected_model_downloads = gr.Textbox(label="downloads", visible=False)
    
    selected_model_search_data = gr.Textbox(label="search_data", visible=True)
    selected_model_hf_data = gr.Textbox(label="hf_data", visible=True)
    selected_model_config_data = gr.Textbox(label="config_data", visible=True)
    gr.Markdown(
        """
        <hr>
        """
    )            

    inp.submit(search_models, inputs=inp, outputs=[model_dropdown]).then(lambda: gr.update(visible=True), None, model_dropdown)
    btn.click(search_models, inputs=inp, outputs=[model_dropdown]).then(lambda: gr.update(visible=True), None, model_dropdown)

    with gr.Row():
        port_model = gr.Number(value=8001,visible=False,label="Port of model: ")
        port_vllm = gr.Number(value=8000,visible=False,label="Port of vLLM: ")
    
    info_textbox = gr.Textbox(value="Interface not possible for selected model. Try another model or check 'pipeline_tag', 'transformers', 'private', 'gated'", show_label=False, visible=False)
    btn_dl = gr.Button("Download", visible=True)
    btn_deploy = gr.Button("Deploy", visible=True)
    
    model_dropdown.change(get_info, model_dropdown, [selected_model_search_data,selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_downloads,selected_model_container_name]).then(get_additional_info, model_dropdown, [selected_model_hf_data, selected_model_config_data, selected_model_id, selected_model_size, selected_model_gated]).then(lambda: gr.update(visible=True), None, selected_model_pipeline_tag).then(lambda: gr.update(visible=True), None, selected_model_transformers).then(lambda: gr.update(visible=True), None, selected_model_private).then(lambda: gr.update(visible=True), None, selected_model_downloads).then(lambda: gr.update(visible=True), None, selected_model_size).then(lambda: gr.update(visible=True), None, selected_model_gated).then(lambda: gr.update(visible=True), None, port_model).then(lambda current_value: current_value + 1, port_model, port_model).then(lambda: gr.update(visible=True), None, port_vllm).then(lambda current_value: current_value + 1, port_vllm, port_vllm).then(gr_load_check, [selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_gated],[info_textbox,btn_dl])

    create_response = gr.Textbox(label="Building container...", show_label=True, visible=False)  
    
    btn_interface = gr.Button("Load Interface",visible=False)
    @gr.render(inputs=[selected_model_pipeline_tag, selected_model_id], triggers=[btn_interface.click])
    def show_split(text_pipeline, text_model):
        if len(text_model) == 0:
            gr.Markdown("Error pipeline_tag or model_id")
        else:
            selected_model_id_arr = str(text_model).split('/')
            print(f'selected_model_id_arr {selected_model_id_arr}...')            
            gr.Interface.from_pipeline(pipeline(text_pipeline, model=f'/models/{selected_model_id_arr[0]}/{selected_model_id_arr[1]}'))



    with gr.Row(visible=True) as engine_arguments_row:

        # max_model_len = gr.Number(label="max_model_len", value=2048, visible=True)
                
        enforce_eager = gr.Checkbox(label="enforce_eager", info="Did they do it in the morning?")     
        tensor_parallel_size = gr.Number(label="tensor_parallel_size", value=1, visible=True)
        gpu_memory_utilization = gr.Textbox(label="gpu_memory_utilization", placeholder="0.87", value=0.87, visible=True)

    max_model_len = gr.Slider(1, 8096, value=2048, label="max_model_len", info="Choose between 2 and 20"),
    engine_args = gr.Button("Show Engine Arguments", scale=0)
    engine_args_close = gr.Button("Close Engine Arguments", scale=0, visible=False)
    
    engine_args.click(
        lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [engine_args,engine_args_close, engine_arguments_row]
    )
    
    engine_args_close.click(
        lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [engine_args,engine_args_close, engine_arguments_row]
    )


    
    def sentence_builder(quantity, animal, countries, place, activity_list, morning):
        return f"""The {quantity} {animal}s from {" and ".join(countries)} went to the {place} where they {" and ".join(activity_list)} until the {"morning" if morning else "night"}"""

    demo = gr.Interface(
        sentence_builder,
        [
            gr.Slider(2, 20, value=4, label="Count", info="Choose between 2 and 20"),
            gr.Dropdown(
                ["cat", "dog", "bird"], label="Animal", info="Will add more animals later!"
            ),
            gr.CheckboxGroup(["USA", "Japan", "Pakistan"], label="Countries", info="Where are they from?"),
            gr.Radio(["park", "zoo", "road"], label="Location", info="Where did they go?"),
            gr.Dropdown(
                ["ran", "swam", "ate", "slept"], value=["swam", "slept"], multiselect=True, label="Activity", info="Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed auctor, nisl eget ultricies aliquam, nunc nisl aliquet nunc, eget aliquam nisl nunc vel nisl."
            ),
            gr.Checkbox(label="Morning", value=True, info="Did they do it in the morning?"),
        ],
        "text"
    )
    
    
    load_btn = gr.Button("Load into vLLM (port: 1370)", visible=True)

    
    
    with gr.Row():
        top_p = gr.Textbox(label="top_p", placeholder="0.95", value=0.95, visible=True)
        temperature = gr.Slider(0.1, 1.0, step=0.1, label="temperature", value=0.8, visible=True)
        max_tokens = gr.Number(label="max_tokens", value=150, visible=True)
    
    
    prompt_in = gr.Textbox(placeholder="Ask a question", value="Follow the", label="Query", show_label=True, visible=True)  
    prompt_out = gr.Textbox(placeholder="Result will appear here", label="Output", show_label=True, visible=True)
    prompt_btn = gr.Button("Submit", visible=True)
    
    # load_btn.click(lambda model, pipeline_tag, max_model_len, tensor_parallel_size, gpu_memory_utilization, top_p, temperature, max_tokens, prompt_in: vllm_api("load", model, pipeline_tag, max_model_len, tensor_parallel_size, gpu_memory_utilization, top_p, temperature, max_tokens, prompt_in), inputs=[model_dropdown, selected_model_pipeline_tag, max_model_len, tensor_parallel_size, gpu_memory_utilization, top_p, temperature, max_tokens, prompt_in], outputs=prompt_out)
        
    # prompt_btn.click(lambda model, pipeline_tag, max_model_len, tensor_parallel_size, gpu_memory_utilization, top_p, temperature, max_tokens, prompt_in: vllm_api("generate", model, pipeline_tag, max_model_len, tensor_parallel_size, gpu_memory_utilization, top_p, temperature, max_tokens, prompt_in), inputs=[model_dropdown, selected_model_pipeline_tag, max_model_len, tensor_parallel_size, gpu_memory_utilization, top_p, temperature, max_tokens, prompt_in], outputs=prompt_out)

    
    gpu_dataframe = gr.Dataframe(label="GPU information")
    gpu_timer = gr.Timer(1,active=True)
    gpu_timer.tick(gpu_to_pd, outputs=gpu_dataframe)
    
    network_dataframe = gr.Dataframe(label="Network information")
    network_timer = gr.Timer(1,active=True)
    network_timer.tick(network_to_pd, outputs=network_dataframe)
    
    container_state = gr.State([])   
    docker_container_list = get_docker_container_list()     
    @gr.render(inputs=container_state)
    def render_container(render_container_list):
        docker_container_list = get_docker_container_list()
        docker_container_list_sys_running = [c for c in docker_container_list if c["State"]["Status"] == "running" and c["Name"] in ["/container_redis","/container_backend", "/container_frontend"]]
        docker_container_list_vllm_running = [c for c in docker_container_list if c["State"]["Status"] == "running" and c["Name"] not in ["/container_redis","/container_backend", "/container_frontend"]]
        docker_container_list_sys_not_running = [c for c in docker_container_list if c["State"]["Status"] != "running" and c["Name"] in ["/container_redis","/container_backend", "/container_frontend"]]
        docker_container_list_vllm_not_running = [c for c in docker_container_list if c["State"]["Status"] != "running" and c["Name"] in ["/container_redis","/container_backend", "/container_frontend"]]

        def refresh_container():
            try:
                global docker_container_list
                response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method": "list"})
                docker_container_list = response.json()
                return docker_container_list
            
            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return f'err {str(e)}'
            
        gr.Markdown(f'### System Container running ({len(docker_container_list_sys_running)})')

        for current_container in docker_container_list_sys_running:
            with gr.Row():
                
                container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                
                container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
    
                container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                
                container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                
            with gr.Row():
                container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)

            with gr.Row():            
                logs_btn = gr.Button("Show Logs", scale=0)
                logs_btn_close = gr.Button("Close Logs", scale=0, visible=False)     
                
                logs_btn.click(
                    docker_api_logs,
                    inputs=[container_id],
                    outputs=[container_log_out]
                ).then(
                    lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [logs_btn,logs_btn_close, container_log_out]
                )
                
                logs_btn_close.click(
                    lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [logs_btn,logs_btn_close, container_log_out]
                )

                stop_btn = gr.Button("Stop", scale=0)
                delete_btn = gr.Button("Delete", scale=0, variant="stop")

                stop_btn.click(
                    docker_api_stop,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )

                delete_btn.click(
                    docker_api_delete,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )
                
            gr.Markdown(
                """
                <hr>
                """
            )


        gr.Markdown(f'### System Container not running ({len(docker_container_list_sys_not_running)})')

        for current_container in docker_container_list_sys_not_running:
            with gr.Row():
                
                container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container ID")
                
                container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
    
                container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                
                container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
            
            with gr.Row():
                container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)
                
            with gr.Row():
                logs_btn = gr.Button("Show Logs", scale=0)
                logs_btn_close = gr.Button("Close Logs", scale=0, visible=False)
                
                logs_btn.click(
                    docker_api_logs,
                    inputs=[container_id],
                    outputs=[container_log_out]
                ).then(
                    lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [logs_btn,logs_btn_close, container_log_out]
                )
                
                logs_btn_close.click(
                    lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [logs_btn,logs_btn_close, container_log_out]
                )
                                
                start_btn = gr.Button("Start", scale=0)
                delete_btn = gr.Button("Delete", scale=0, variant="stop")

                start_btn.click(
                    docker_api_start,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )

                delete_btn.click(
                    docker_api_delete,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )
            
            gr.Markdown(
                """
                <hr>
                """
            )
            
             
        gr.Markdown(f'### vLLM Container running ({len(docker_container_list_vllm_running)})')

        for current_container in docker_container_list_vllm_running:
            with gr.Row():
                
                container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                
                container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
    
                container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                
                container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                
            with gr.Row():
                container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)

            with gr.Row():            
                btn_logs_file_open = gr.Button("Show Log File", scale=0)
                btn_logs_file_close = gr.Button("Close Log File", scale=0, visible=False)   
                btn_logs_docker_open = gr.Button("Show Docker Logs", scale=0)
                btn_logs_docker_close = gr.Button("Close Docker Logs", scale=0, visible=False)     
                
                
                
                btn_logs_file_open.click(
                    load_log_file,
                    inputs=[container_id],
                    outputs=[container_log_out]
                ).then(
                    lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                )
                
                btn_logs_file_close.click(
                    lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_file_open,btn_logs_file_close, container_log_out]
                )
                
                btn_logs_docker_open.click(
                    docker_api_logs,
                    inputs=[container_id],
                    outputs=[container_log_out]
                ).then(
                    lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                )
                
                btn_logs_docker_close.click(
                    lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [btn_logs_docker_open,btn_logs_docker_close, container_log_out]
                )
                
                stop_btn = gr.Button("Stop", scale=0)
                delete_btn = gr.Button("Delete", scale=0, variant="stop")

                stop_btn.click(
                    docker_api_stop,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )

                delete_btn.click(
                    docker_api_delete,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )
                
            gr.Markdown(
                """
                <hr>
                """
            )


        gr.Markdown(f'### vLLM Container not running ({len(docker_container_list_vllm_not_running)})')

        for current_container in docker_container_list_vllm_not_running:
            with gr.Row():
                
                container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container ID")
                
                container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
    
                container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                
                container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
            
            with gr.Row():
                container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)
                
            with gr.Row():
                logs_btn = gr.Button("Show Logs", scale=0)
                logs_btn_close = gr.Button("Close Logs", scale=0, visible=False)
                
                logs_btn.click(
                    docker_api_logs,
                    inputs=[container_id],
                    outputs=[container_log_out]
                ).then(
                    lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [logs_btn,logs_btn_close, container_log_out]
                )
                
                logs_btn_close.click(
                    lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [logs_btn,logs_btn_close, container_log_out]
                )
                                
                start_btn = gr.Button("Start", scale=0)
                delete_btn = gr.Button("Delete", scale=0, variant="stop")

                start_btn.click(
                    docker_api_start,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )

                delete_btn.click(
                    docker_api_delete,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )
            
            gr.Markdown(
                """
                <hr>
                """
            )
            
    def refresh_container_list():
        try:
            global docker_container_list
            response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method": "list"})
            docker_container_list = response.json()
            return docker_container_list
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return f'err {str(e)}'

    timer_dl = gr.Timer(1,active=False)
    timer_dl.tick(get_download_speed, outputs=create_response)    
    
    
    timer_c = gr.Timer(1,active=False)
    timer_c.tick(refresh_container)
    
    btn_dl.click(lambda: gr.update(label="Starting download ...",visible=True), None, create_response).then(lambda: gr.Timer(active=True), None, timer_dl).then(download_from_hf_hub, model_dropdown, create_response).then(lambda: gr.Timer(active=False), None, timer_dl).then(lambda: gr.update(label="Download finished!"), None, create_response).then(lambda: gr.update(visible=True), None, btn_interface)

    
    btn_deploy.click(lambda: gr.update(label="Building vLLM container",visible=True), None, create_response).then(docker_api_create,inputs=[model_dropdown,selected_model_pipeline_tag,port_model,port_vllm],outputs=create_response).then(refresh_container, outputs=[container_state]).then(lambda: gr.Timer(active=True), None, timer_dl).then(lambda: gr.update(visible=True), None, btn_interface)


    
    

app.launch(server_name=f'{os.getenv("FRONTEND_IP")}', server_port=int(os.getenv("FRONTEND_PORT")))

