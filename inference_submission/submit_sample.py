import os
import subprocess

submit_name = "test_set"

# subprocess.run(['python',
# '/home/guo_chen2023/LLM-ToM-GenAI/inference_submission/inference.py',
# "-o",
# f"/home/guo_chen2023/LLM-ToM-GenAI/outputs/qwen_{submit_name}"])


subprocess.run(['python',
'/home/guo_chen2023/LLM-ToM-GenAI/inference_submission/inference.py',
"-o",
f"/home/guo_chen2023/LLM-ToM-GenAI/outputs/qwen_{submit_name}",
"-c",
"/home/guo_chen2023/LLM-ToM-GenAI/model_gardem_configs/qwen.yaml",
"-d",
"/home/guo_chen2023/LLM-ToM-GenAI/test.csv"])