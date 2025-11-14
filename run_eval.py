# fantom_eval.py
# -*- coding: utf-8 -*-

import os
import csv
import json
from typing import List, Dict, Tuple

import yaml
from dotenv import load_dotenv
from google import genai
from google.cloud import storage
import requests
import re
from collections import Counter

F1_THRESHOLD = 0.5  # ablation study



def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_client() -> genai.Client:
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )
    return client


def _load_from_gcs(gs_path: str):
    """
    gs://bucket/path/to/file.json
    """
    assert gs_path.startswith("gs://")
    inner = gs_path[len("gs://"):]
    bucket_name, blob_path = inner.split("/", 1)

    gcs_client = storage.Client()
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    text = blob.download_as_text()
    return json.loads(text)


def _load_from_http(url: str):
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def _load_from_local(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_fantom(fantom_path: str) -> List[Dict]:
    print(f"[FANTOM] Loading dataset from: {fantom_path}")

    if fantom_path.startswith("gs://"):
        data = _load_from_gcs(fantom_path)
    elif fantom_path.startswith("http://") or fantom_path.startswith("https://"):
        data = _load_from_http(fantom_path)
    else:
        data = _load_from_local(fantom_path)

    if isinstance(data, list):
        examples = data
    elif isinstance(data, dict) and "data" in data:
        examples = data["data"]
    else:
        raise ValueError("Unexpected fantom_v1.json format (not list / no 'data' key)")

    print(f"[FANTOM] Loaded {len(examples)} entries")
    return examples

# factQ

def get_context(example: Dict, context_field: str) -> str:
    ctx = example.get(context_field)
    if not ctx:
        ctx = example.get("full_context", "")
    return str(ctx)


def _pick_fact_qa_block(example: Dict) -> Dict:
    fact_block = example.get("factQA")
    if fact_block is None:
        raise KeyError("Example has no 'factQA' field")

    if isinstance(fact_block, list):
        if not fact_block:
            raise ValueError("factQA list is empty")
        fact_block = fact_block[0]

    if not isinstance(fact_block, dict):
        raise ValueError("factQA is neither dict nor list[dict]")

    return fact_block


def build_factq_prompt_and_gold(
    example: Dict,
    context_field: str = "full_context",
) -> Tuple[str, str, Dict]:
    context = get_context(example, context_field)
    fact = _pick_fact_qa_block(example)
    question = (
        fact.get("question")
        or fact.get("fact_question")
        or fact.get("q")
    )
    if question is None:
        raise KeyError("Cannot find question field in factQA block; please adjust key names.")
    
    gold = (
        fact.get("correct_answer")
        or fact.get("full_answer")
        or fact.get("full_fact")
        or fact.get("answer")
        or fact.get("gold_answer")
    )
    if gold is None:
        raise KeyError("Cannot find gold answer field in factQA block; please adjust key names.")

    question = str(question).strip()
    gold = str(gold).strip()

    prompt = f"""You are evaluating a Theory-of-Mind benchmark called FanToM.

Read the following multi-party casual conversation carefully, then answer a factual question about it.
Use only information stated in the conversation. Do not make up extra facts.

Conversation:
{context}

Question:
{question}

Answer the question as concisely as possible, in one short sentence.
Do NOT include explanations, just the answer itself.
"""

    meta = {
        "set_id": example.get("set_id"),
        "part_id": example.get("part_id"),
        "conv_id": example.get("conv_id"),
        "qa_type": "FactQ",
    }

    return prompt, gold, meta


# 如果之后你想扩展到 BeliefQ / Answerability / InfoAccess，可以按类似风格再写：
#
# def build_belief_dist_prompt_and_gold(...):
#     ...
#
# def build_answerability_list_prompt_and_gold(...):
#     ...
#
# 然后在 run_eval 里根据 cfg["eval"]["task_type"] 做分派。


# Gemini

def call_model(client: genai.Client, model_name: str, prompt: str) -> str:
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return (resp.text or "").strip()


def normalize_text(s: str) -> str:
    return s.strip().lower()

def simple_tokenize(text: str):
    """
    非严格分词：小写 + 去标点 + 按空格切分
    """
    text = normalize_text(text)
    # 把标点变成空格
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    return tokens


def token_f1(pred: str, gold: str) -> float:
    """
    token-level F1。
    这里实现一个标准版本：用多重集交集计算 overlap。
    """
    pred_tokens = simple_tokenize(pred)
    gold_tokens = simple_tokenize(gold)

    if not pred_tokens or not gold_tokens:
        return 0.0

    p_counts = Counter(pred_tokens)
    g_counts = Counter(gold_tokens)

    overlap = 0
    for t, c in p_counts.items():
        overlap += min(c, g_counts.get(t, 0))

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)




def run_eval(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]

    fantom_path = data_cfg["fantom_url"]
    context_field = data_cfg.get("context_field", "full_context")

    task_type = eval_cfg.get("task_type", "FactQ")
    max_examples = eval_cfg.get("max_examples")
    model_name = eval_cfg["model_name"]
    output_path = eval_cfg.get("output_path", f"fantom_{task_type}_results.csv")

    client = init_client()

    examples = load_fantom(fantom_path)
    if max_examples is not None:
        examples = examples[:max_examples]
        print(f"[FANTOM] Subsampled to first {len(examples)} examples.")

    # 目前只实现 FactQ
    if task_type != "FactQ":
        raise NotImplementedError(f"task_type={task_type} only supports 'FactQ' for now.")

    results = []
    num_correct = 0        # F1 >= 阈值 的个数
    total_f1 = 0.0         # 用来算 mean F1
    total_q = 0            # 有效问题数


    for idx, ex in enumerate(examples, start=1):
        try:
            prompt, gold, meta = build_factq_prompt_and_gold(
                ex,
                context_field=context_field,
            )
        except Exception as e:
            print(f"[WARN] Skip example {idx} due to mapping error: {e}")
            continue

        pred = call_model(client, model_name, prompt)

        total_q += 1

        if gold:
            f1 = token_f1(pred, gold)
        else:
            f1 = 0.0

        total_f1 += f1

        # 可选：用 F1 阈值当成一个粗略正确/错误指标
        is_correct = (f1 >= F1_THRESHOLD) if gold else False

        if is_correct:
            num_correct += 1

        row = {
            "global_index": idx,
            "question_index": total_q,
            "set_id": meta.get("set_id"),
            "part_id": meta.get("part_id"),
            "conv_id": meta.get("conv_id"),
            "qa_type": meta.get("qa_type"),
            "gold": gold,
            "prediction": pred,
            "f1": f1,
            "correct": int(is_correct),
        }
        results.append(row)

        if idx % 10 == 0:
            print(f"[FANTOM] Processed {idx} / {len(examples)} entries "
                  f"({total_q} valid FactQ)")

    if total_q > 0:
        mean_f1 = total_f1 / total_q
        acc = num_correct / total_q
        print(f"\n[FANTOM] FactQ mean token F1: {mean_f1:.4f}")
        print(f"[FANTOM] FactQ accuracy (F1 >= {F1_THRESHOLD}): "
              f"{acc:.4f} ({num_correct}/{total_q})")
    else:
        print("\n[FANTOM] No valid FactQ examples were processed. "
              "Please check the factQA field names.")


    fieldnames = [
        "global_index",
        "question_index",
        "set_id",
        "part_id",
        "conv_id",
        "qa_type",
        "gold",
        "prediction",
        "f1",
        "correct",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[FANTOM] Saved results to {output_path}")


if __name__ == "__main__":
    run_eval("config.yaml")
