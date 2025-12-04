import os
import subprocess

csv_dir = "/home/guo_chen2023/LLM-ToM-GenAI/tombench_csvs"

for c_file in os.listdir(csv_dir):
    submit_name = c_file.split('.')[0].lower().replace(' ', '_')

    if not os.path.exists(f"/home/guo_chen2023/output_qwen_{submit_name}"):
        subprocess.run(['python',
        '/home/guo_chen2023/LLM-ToM-GenAI/inference_submission/inference.py',
        "-d",
        f"{csv_dir}/{c_file}",
        "-o",
        f"/home/guo_chen2023/output_qwen_{submit_name}"])