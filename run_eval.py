import os
import yaml
import requests
import json
from dotenv import load_dotenv
from google import genai

# ========= 加载 .env =========
load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION   = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# ========= 加载 YAML 配置 =========
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

FANTOM_URL   = cfg["data"]["fantom_url"]
MAX_EXAMPLES = cfg["eval"]["max_examples"]
MODEL_NAME   = cfg["eval"]["model_name"]

# ========= 初始化 Gemini Client =========
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

# ========= 读取 FANTOM =========
def load_fantom():
    r = requests.get(FANTOM_URL)
    r.raise_for_status()
    data = json.loads(r.text)

    # 这里还是之前那句：具体结构你打开文件看一下再改 key
    if isinstance(data, list):
        examples = data
    elif isinstance(data, dict):
        examples = data.get("data", [])
    else:
        raise ValueError("Unexpected fantom_v1.json format")

    print(f"Loaded {len(examples)} examples from FANTOM via {FANTOM_URL}")
    return examples

# 后面就是你之前的 eval 循环逻辑...
