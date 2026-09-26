"""Serve a local Laya checkpoint with CUDA; no model downloads at runtime."""
import argparse
import ipaddress
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "host": "127.0.0.1",
    "port": 8002,
    "allowed_clients": ["127.0.0.1"],
    "model_path": "runtime/models/multilingual",
    "key_file": "runtime/private/gpu-api-key",
}


def concrete_ip(value):
    address = ipaddress.ip_address(value)
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    if address.is_unspecified or address.is_multicast or str(address) == "255.255.255.255":
        raise ValueError("Use a specific unicast IP address, never a wildcard or subnet")
    return str(address)


def load_config(path):
    supplied = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("Invalid GPU configuration fields")
    config = DEFAULTS | supplied
    if not isinstance(config["host"], str):
        raise ValueError("host must be a specific IP address")
    config["host"] = concrete_ip(config["host"])
    if type(config["port"]) is not int or not 1 <= config["port"] <= 65535:
        raise ValueError("port must be an integer from 1 to 65535")
    clients = config["allowed_clients"]
    if not isinstance(clients, list) or not clients or any(not isinstance(client, str) for client in clients):
        raise ValueError("allowed_clients must contain specific trusted IP addresses")
    config["allowed_clients"] = sorted({concrete_ip(client) for client in clients})
    for field in ("model_path", "key_file"):
        if not isinstance(config[field], str) or not config[field].strip():
            raise ValueError(f"{field} must be a nonempty path")
        config[field] = str((ROOT / config[field]).resolve())
    return config


def read_key(path):
    key = Path(path).read_text(encoding="utf-8").strip()
    if len(key) < 24:
        raise ValueError("The GPU API key must be random and at least 24 characters long")
    return key


def serve(config):
    key = read_key(config["key_file"])
    if not Path(config["model_path"]).is_dir():
        raise ValueError("The configured model directory does not exist")
    os.environ.update(TORCH_DISABLE_NATIVE_JIT="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", LAYA_DEVICE="cuda", LAYA_API_KEY=key)

    import torch
    import uvicorn
    from fastapi.responses import JSONResponse
    from laya.router import Router
    from laya.serve import create_app

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the GPU service was not started")
    torch.set_num_threads(4)
    router = Router(models={"multilingual": config["model_path"]}, device="cuda",
                    default="multilingual", max_loaded=1, auto_task_detection=False)
    agent = router.load("multilingual")
    if agent.device.type != "cuda" or torch.cuda.memory_allocated() == 0:
        raise RuntimeError("Laya did not load the model onto the GPU")
    print("Laya GPU ready:", torch.cuda.get_device_name(), flush=True)
    app = create_app(router)
    allowed = set(config["allowed_clients"])

    @app.middleware("http")
    async def trusted_clients(request, call_next):
        try:
            client = concrete_ip(request.client.host) if request.client else None
        except ValueError:
            client = None
        if client not in allowed:
            return JSONResponse({"error": "Access denied"}, status_code=403)
        return await call_next(request)

    uvicorn.run(app, host=config["host"], port=config["port"], access_log=False,
                proxy_headers=False, log_level="warning")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.local.json"))
    parser.add_argument("--model-path", type=str, default=None, help="Override configured model path")
    parser.add_argument("--print-config", action="store_true", help="Validate config without loading CUDA or secrets")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.model_path:
            config["model_path"] = str(Path(args.model_path).resolve())
        if args.print_config:
            print(json.dumps(config))
        else:
            serve(config)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"GPU service: {error}\n")


if __name__ == "__main__":
    main()
