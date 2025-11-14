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


# ==================== 配置 & Client ====================

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_client() -> genai.Client:
    """
    使用本地 GOOGLE_APPLICATION_CREDENTIALS / gcloud login 的 ADC，
    再从 .env 中拿 project / location.
    """
    load_dotenv()
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )
    return client


# ==================== 读取 FANTOM 数据 ====================

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
    """
    支持三种方式：
      - gs://...      → google-cloud-storage
      - http(s)://... → requests
      - 其它          → 当成本地文件路径
    所有路径都只从 config.yaml 里读，不在代码里写死。
    """
    print(f"[FANTOM] Loading dataset from: {fantom_path}")

    if fantom_path.startswith("gs://"):
        data = _load_from_gcs(fantom_path)
    elif fantom_path.startswith("http://") or fantom_path.startswith("https://"):
        data = _load_from_http(fantom_path)
    else:
        data = _load_from_local(fantom_path)

    # 统一成 list[dict]
    if isinstance(data, list):
        examples = data
    elif isinstance(data, dict) and "data" in data:
        examples = data["data"]
    else:
        raise ValueError("Unexpected fantom_v1.json format (not list / no 'data' key)")

    print(f"[FANTOM] Loaded {len(examples)} entries")
    return examples


# ==================== FanToM: prompt & gold 映射 ====================
# 当前先实现 FACTQ，一条 conversation 取对应的 factQA 问题。
# 你给的 schema:
# ['set_id', 'part_id', 'conv_id', 'full_context', 'short_context',
#  'missed_info', 'joining_speaker', 'factQA', 'beliefQAs',
#  'infoAccessibilityQA_list', 'answerabilityQA_list',
#  'infoAccessibilityQAs_binary', 'answerabilityQAs_binary']

def get_context(example: Dict, context_field: str) -> str:
    """
    从 full_context / short_context 里挑一个作为输入。
    """
    ctx = example.get(context_field)
    if not ctx:
        # 兜底：没有这个 key 就退回 full_context
        ctx = example.get("full_context", "")
    return str(ctx)


def _pick_fact_qa_block(example: Dict) -> Dict:
    """
    从样本里拿到一个 FactQ block。
    FanToM 原始构造是“每个 FactQ 再派生 6 种 ToM QA”，
    这里先只用 FactQ 做基本 comprehension。
    
    你需要根据实际 JSON 确认 factQA 的结构：
      - 如果 factQA 是 dict： {question: ..., full_answer: ...}
      - 如果 factQA 是 list： [ {question: ..., ...}, ... ]
    下面的代码默认：
      1) 如果是 list，就用第一个；
      2) 字段名先假设为 "question" / "answer" / "full_answer" 之类，
         你可以跑一下 print 看再改。
    """
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
    """
    构造一个 FACTQ 的 prompt 和 gold。
    返回： (prompt_text, gold_answer_str, extra_meta)
    """
    context = get_context(example, context_field)
    fact = _pick_fact_qa_block(example)

    # ========= 这里请你确认一下字段名 =========
    # 建议你先 print(fact.keys()) 看一眼：
    #   - 常见写法可能是："question", "full_answer", "limited_answer", ...
    # 下面先按最朴素的 guess 写：question / answer / full_answer，供你微调。
    question = (
        fact.get("question")
        or fact.get("fact_question")
        or fact.get("q")
    )
    if question is None:
        raise KeyError("Cannot find question field in factQA block; please adjust key names.")

    # FULL FACT A 是"上帝视角的真实答案"，用来衡量 basic comprehension
    # Fantom 数据集使用 correct_answer 字段
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

    # ========= Prompt：对话 + 问题 =========
    # 按论文设定：模型看到整段对话，扮演“全知观察者”回答 fact 问题
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


# ==================== 调用 Gemini ====================

def call_model(client: genai.Client, model_name: str, prompt: str) -> str:
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return (resp.text or "").strip()


def normalize_text(s: str) -> str:
    return s.strip().lower()


# ==================== 主评测循环 ====================

def run_eval(config_path: str = "config.yaml"):
    # 1. 读配置
    cfg = load_config(config_path)

    data_cfg = cfg["data"]
    eval_cfg = cfg["eval"]

    fantom_path = data_cfg["fantom_url"]
    context_field = data_cfg.get("context_field", "full_context")

    task_type = eval_cfg.get("task_type", "FactQ")
    max_examples = eval_cfg.get("max_examples")
    model_name = eval_cfg["model_name"]
    output_path = eval_cfg.get("output_path", f"fantom_{task_type}_results.csv")

    # 2. 初始化 Gemini client
    client = init_client()

    # 3. 载入数据
    examples = load_fantom(fantom_path)
    if max_examples is not None:
        examples = examples[:max_examples]
        print(f"[FANTOM] Subsampled to first {len(examples)} examples.")

    # 4. 逐条跑模型（目前先只支持 FactQ）
    if task_type != "FactQ":
        raise NotImplementedError(f"task_type={task_type} 暂时只实现了 FactQ，其他题型可以仿照再写。")

    results = []
    num_correct = 0
    total_q = 0

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
        is_correct = False
        if gold:
            # 这里只是最 naive 的 string match；
            # 真正要和论文对齐得上 Sentence-BERT + token-F1 
            is_correct = normalize_text(pred) == normalize_text(gold)

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
            "correct": int(is_correct),
        }
        results.append(row)

        if idx % 10 == 0:
            print(f"[FANTOM] Processed {idx} / {len(examples)} entries "
                  f"({total_q} valid FactQ)")

    # 5. 打印简单指标
    if total_q > 0:
        acc = num_correct / total_q
        print(f"\n[FANTOM] FactQ accuracy (exact string match): "
              f"{acc:.4f} ({num_correct}/{total_q})")
    else:
        print("\n[FANTOM] No valid FactQ examples were processed. "
              "请检查 factQA 的字段名。")

    # 6. 写出 CSV 结果
    fieldnames = [
        "global_index",
        "question_index",
        "set_id",
        "part_id",
        "conv_id",
        "qa_type",
        "gold",
        "prediction",
        "correct",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[FANTOM] Saved results to {output_path}")


if __name__ == "__main__":
    run_eval("config.yaml")
