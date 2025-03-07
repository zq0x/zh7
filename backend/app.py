from fastapi import FastAPI, Request, HTTPException
import json
import subprocess
import docker
from docker.types import DeviceRequest
import time
import os
import redis.asyncio as redis
import sys
from fastapi.responses import JSONResponse
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
import pynvml
import psutil


print(f'** connecting to redis on port: {os.getenv("REDIS_PORT")} ... ')
r = redis.Redis(host="redis", port=int(os.getenv("REDIS_PORT", 6379)), db=0)

print(f'** connecting to pynvml ... ')
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()
print(f'** pynvml found GPU: {device_count}')

device_uuids = []
for i in range(0,device_count):
    print(f'1 i {i}')
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    print(f'1 handle {handle}')
    current_uuid = pynvml.nvmlDeviceGetUUID(handle)
    device_uuids.append(current_uuid)

print(f'** pynvml found uuids ({len(device_uuids)}): {device_uuids} ')



# def get_vllm_info():
#     container_info = []
#     try:        
#         res_container_list = client.containers.list(all=True)
#         print(f'** [get_container_info] {res_container_list} ({len(res_container_list)})')
#         for res_container_i in range(0,len(res_container_list)):
#             print("res_container_i")
#             print(res_container_i)
#             print("res_container_list[res_container_i]")
#             print(res_container_list[res_container_i])
#             container_info.append({
#                     "container_i": f'{res_container_i}',
#                     "container_info": f'{res_container_i}'
#             })
#         return container_info
#     except Exception as e:
#         print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
#         return container_info


# async def redis_timer_vllm():
#     while True:
#         try:
#             current_container_info = get_container_info()
#             res_db_container = await r.get('db_container')
#             if res_db_container is not None:
#                 db_container = json.loads(res_db_container)
#                 updated_container_data = []
#                 print(f' [container] 1 len(current_gpu_info): {len(current_container_info)}')
#                 for container_i in range(0,len(current_container_info)):
#                     print(f' [container] gpu_i: {container_i}')
#                     update_data = {
#                         "container_i": container_i,
#                         "container_info": str(current_container_info[container_i]),
#                         "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#                     }
#                     updated_container_data.append(update_data)
#                     print(f'[container] 1 updated_container_data: {updated_container_data}')
#                 await r.set('db_container', json.dumps(updated_container_data))
#             else:
#                 updated_container_data = []
#                 for container_i in range(0,len(current_container_info)):
#                     update_data = {
#                         "container_i": container_i,
#                         "container_info": str(current_container_info[container_i]),
#                         "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#                     }
#                     updated_container_data.append(update_data)
#                     print(f'[container] 2 updated_container_data: {updated_container_data}')
#                 await r.set('db_container', json.dumps(updated_container_data))
#             await asyncio.sleep(1.0)
#         except Exception as e:
#             print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
#             await asyncio.sleep(1.0)
















rx_change_arr = []
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





prev_bytes_recv = 0

def get_network_info():
    network_info = []
    try: 
            
        current_total_dl = get_download_speed()
        network_info.append({
            "container": f'all',
            "info": f'network_info_blank',            
            "current_dl": f'{current_total_dl}',
            "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        })
        
        print(f'finding all containers ..')
        
        res_container_list = client.containers.list(all=True)
        print(f'found {len(res_container_list)} containers!')
        
        for c in res_container_list:
            print('c')
            print(c)
            print('c["name"]')
            print(c["name"])
            
            stats = c.stats(stream=False)
            print('stats')
            print(stats)
            network_info.append({
                "container": f'{c["name"]}',
                "info": f'network_info_blank', 
                "current_dl": f'000000000000000',
                "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            })
            
        print(f'got all containers! printing final before responding ')
        print('network_info')
        print(network_info)           
        return network_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return network_info

async def redis_timer_network():
    while True:
        try:
            current_network_info = get_network_info()
            res_db_network = await r.get('db_network')
            if res_db_network is not None:
                db_network = json.loads(res_db_network)
                updated_network_data = []
                print(f' [network] 1 len(current_gpu_info): {len(current_network_info)}')
                for net_info_obj in current_network_info:
                    print(f' [network] net_info_obj: {net_info_obj}')
                    update_data = {
                        "container": str(net_info_obj["container"]),
                        "info": str(net_info_obj["container"]),
                        "current_dl": str(net_info_obj["current_dl"]),
                        "timestamp": str(net_info_obj["timestamp"]),
                    }
                    updated_network_data.append(update_data)
                    print(f'[network] 1 updated_network_data: {updated_network_data}')
                await r.set('db_network', json.dumps(updated_network_data))
            else:
                updated_network_data = []
                for net_info_obj in current_network_info:
                    update_data = {
                        "container": str(net_info_obj["container"]),
                        "info": str(net_info_obj["container"]),
                        "current_dl": str(net_info_obj["current_dl"]),
                        "timestamp": str(net_info_obj["timestamp"]),
                    }
                    updated_network_data.append(update_data)
                    print(f'[network] 2 updated_network_data: {updated_network_data}')
                await r.set('db_network', json.dumps(updated_network_data))
            await asyncio.sleep(1.0)
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
            await asyncio.sleep(1.0)



pynvml.nvmlInit()
def get_gpu_info():
    try:

        device_count = pynvml.nvmlDeviceGetCount()
        print(f'hm ok device_count: {device_count}')
        gpu_info = []
        for i in range(0,device_count):
            print(f'hm ok i: {i}')
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            current_uuid = pynvml.nvmlDeviceGetUUID(handle)
            print(f'hm ok current_uuid: {current_uuid}')
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = memory_info.used / 1024**2
            mem_total = memory_info.total / 1024**2
            mem_util = (mem_used / mem_total) * 100

            gpu_info.append({
                    "gpu_i": i,
                    "current_uuid": current_uuid,
                    "gpu_util": float(gpu_util),
                    "mem_used": float(mem_used),
                    "mem_total": float(mem_total),
                    "mem_util": float(mem_util)
            })
        print("gpu_info")
        print(gpu_info) 
        return gpu_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return 0

current_gpu_info = get_gpu_info()

async def redis_timer_gpu():
    while True:
        try:
            current_gpu_info = get_gpu_info()
            res_db_gpu = await r.get('db_gpu')
            if res_db_gpu is not None:
                db_gpu = json.loads(res_db_gpu)
                updated_gpu_data = []
                print(f'huhh  len(current_gpu_info): {len(current_gpu_info)}')
                for gpu_i in range(0,len(current_gpu_info)):
                    print(f'huhh  gpu_i: {gpu_i}')
                    update_data = {
                        "gpu_i": gpu_i,
                        "gpu_info": str(current_gpu_info[gpu_i]),
                        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    }
                    updated_gpu_data.append(update_data)
                    print(f'1 updated_gpu_data: {updated_gpu_data}')
                await r.set('db_gpu', json.dumps(updated_gpu_data))
            else:
                updated_gpu_data = []
                for gpu_i in range(0,len(current_gpu_info)):
                    update_data = {
                        "gpu_i": gpu_i,
                        "gpu_info": str(current_gpu_info[gpu_i]),
                        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    }
                    updated_gpu_data.append(update_data)
                    print(f'2 updated_gpu_data: {updated_gpu_data}')
                await r.set('db_gpu', json.dumps(updated_gpu_data))
            await asyncio.sleep(2.0)
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
            await asyncio.sleep(2.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(redis_timer_gpu())
    asyncio.create_task(redis_timer_network())
    yield

app = FastAPI(lifespan=lifespan)
client = docker.from_env()
device_request = DeviceRequest(count=-1, capabilities=[["gpu"]])

@app.get("/")
async def root():
    return f'Hello from server {os.getenv("CONTAINER_PORT")}!'

@app.post("/dockerrest")
async def docker_rest(request: Request):
    try:
        req_data = await request.json()
                
        if req_data["req_method"] == "test":
            print(f'got test!')
            print("req_data")
            print(req_data)
            return JSONResponse({"result": 200, "result_data": "teeest okoko"})
                
        if req_data["req_method"] == "logs":
            req_container = client.containers.get(req_data["req_model"])
            res_logs = req_container.logs()
            res_logs_str = res_logs.decode('utf-8')
            return JSONResponse({"result": 200, "result_data": res_logs_str})

        if req_data["req_method"] == "network":
            req_container = client.containers.get(req_data["req_container_name"])
            stats = req_container.stats(stream=False)
            return JSONResponse({"result": 200, "result_data": stats})

        if req_data["req_method"] == "list":
            res_container_list = client.containers.list(all=True)
            return JSONResponse([container.attrs for container in res_container_list])

        if req_data["req_method"] == "delete":
            req_container = client.containers.get(req_data["req_model"])
            req_container.stop()
            req_container.remove(force=True)
            return JSONResponse({"result": 200})

        if req_data["req_method"] == "stop":
            req_container = client.containers.get(req_data["req_model"])
            req_container.stop()
            return JSONResponse({"result": 200})

        if req_data["req_method"] == "start":
            req_container = client.containers.get(req_data["req_model"])
            req_container.start()
            return JSONResponse({"result": 200})

        if req_data["req_method"] == "create":
            try:
                container_name = str(req_data["req_model"]).replace('/', '_')
                container_name = f'vllm_{container_name}'
                res_db_gpu = await r.get('db_gpu')
                if res_db_gpu is not None:
                    db_gpu = json.loads(res_db_gpu)                    
                                        
                    # check if model already downloaded/downloading
                    all_used_models = [g["used_models"] for g in db_gpu]
                    print(f'all_used_models {all_used_models}')
                    if req_data["req_model"] in all_used_models:
                        return JSONResponse({"result": 302, "result_data": "Model already downloaded. Trying to start container ..."})
                    
                    # check if ports already used
                    all_used_ports = [g["used_ports"] for g in db_gpu]
                    print(f'all_used_ports {all_used_ports}')
                    if req_data["req_port_vllm"] in all_used_ports or req_data["req_port_model"] in all_used_ports:
                        return JSONResponse({"result": 409, "result_data": "Error: Port already in use"})
                    
                    # check if memory available
                    current_gpu_info = get_gpu_info()
                    if current_gpu_info[0]["mem_util"] > 50:
                        all_running_models = [g["running_model"] for g in db_gpu]
                        print(f'all_running_models {all_running_models}')
                        for running_model in all_running_models:
                            req_container = client.containers.get(req_data["req_model"])
                            req_container.stop()
                        
                    # wait for containers to stop
                    for i in range(10):
                        current_gpu_info = get_gpu_info()
                        if current_gpu_info[0]["mem_util"] <= 80:
                            continue
                        else:
                            if i == 9:
                                return JSONResponse({"result": 500, "result_data": "Error: Memory > 80%"})
                            else:
                                time.sleep(1)
                    
                    # get all used ports
                    all_used_ports += [req_data["req_port_vllm"],req_data["req_port_model"]]
                    all_used_models += [req_data["req_port_model"]]
                    add_data = {
                        "gpu": 0, 
                        "gpu_info": "0",
                        "running_model": str(container_name),
                        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        "port_vllm": req_data["req_port_vllm"],
                        "port_model": req_data["req_port_model"],
                        "used_ports": str(all_used_ports),
                        "used_models": str(all_used_models)
                    }
                    
                    db_gpu += [add_data]
                    await r.set('db_gpu', json.dumps(db_gpu))                
                
                else:
                    add_data = {
                        "gpu": 0, 
                        "gpu_info": "0",
                        "running_model": str(container_name),
                        "timestamp": str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        "port_vllm": str(req_data["req_port_vllm"]),
                        "port_model": str(req_data["req_port_model"]),
                        "used_ports": f'{str(req_data["req_port_vllm"])},{str(req_data["req_port_model"])}',
                        "used_models": str(str(req_data["req_model"]))
                    }
                    await r.set('db_gpu', json.dumps(add_data))
                        
                print(f'finding containers to stop to free GPU memory...')
                container_list = client.containers.list(all=True)
                print(f'found total containers: {len(container_list)}')
                # docker_container_list = get_docker_container_list()
                # docker_container_list_running = [c for c in docker_container_list if c["State"]["Status"] == "running"]
                
                # res_container_list = client.containers.list(all=True)
                # return JSONResponse([container.attrs for container in res_container_list])
                
                print(f'mhmmhmhmh')
                vllm_containers_running = [c for c in container_list if c.name.startswith("vllm") and c.status == "running"]
                print(f'found total vLLM running containers: {len(vllm_containers_running)}')
                while len(vllm_containers_running) > 0:
                    print(f'stopping all vLLM containers...')
                    for vllm_container in vllm_containers_running:
                        print(f'stopping container {vllm_container.name}...')
                        vllm_container.stop()
                        vllm_container.wait()
                    print(f'waiting for containers to stop...')
                    time.sleep(2)
                    vllm_containers_running = [c for c in container_list if c.name.startswith("vllm") and c.status == "running"]
                print(f'all vLLM containers stopped successfully') 
                                
                res_container = client.containers.run(
                    "vllm/vllm-openai:latest",
                    command=f'--model {req_data["req_model"]} --tensor-parallel-size 2',
                    name=container_name,
                    runtime=req_data["req_runtime"],
                    volumes={"/home/cloud/.cache/huggingface": {"bind": "/root/.cache/huggingface", "mode": "rw"}},
                    ports={
                        f'{req_data["req_port_vllm"]}/tcp': ("0.0.0.0", req_data["req_port_model"])
                    },
                    ipc_mode="host",
                    device_requests=[device_request],
                    detach=True
                )
                container_id = res_container.id
                return JSONResponse({"result": 200, "result_data": str(container_id)})

            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                r.delete(f'running_model:{str(req_data["req_model"])}')
                return JSONResponse({"result": 404, "result_data": str(e)})

    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result": 500, "result_data": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("CONTAINER_PORT")))