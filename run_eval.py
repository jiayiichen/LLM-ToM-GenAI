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


def _normalize_name_list(gold_raw):
    """
    把 gold 里面的角色名单统一成一个逗号分隔的字符串，方便后面做 F1。
    支持几种常见情况：
      - list[str]
      - list[dict{name: ...}]
      - 单个字符串
    """
    if isinstance(gold_raw, list):
        if gold_raw and isinstance(gold_raw[0], dict):
            names = [x.get("name", "") for x in gold_raw]
        else:
            names = [str(x) for x in gold_raw]
        names = [n.strip() for n in names if n and n.strip()]
        # 排一下序，减少顺序带来的 diff
        names = sorted(set(names), key=str.lower)
        return ", ".join(names)
    else:
        return str(gold_raw).strip()


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

# beliefQ

def build_belief_dist_prompt_and_gold(
    example: dict,
    context_field: str = "short_context",
):
    """
    BeliefQ[Dist]（开放式）：
    - 输入：对话 + 角色的 belief 问题
    - gold：角色视角的 correct_answer
    - wrong_answer：上帝视角 / 错误信念，用于之后做“更像 belief 还是 fact”的对比

    当前 FanToM beliefQAs 的结构（你打印出来的）是：
        {
            "question": "...",
            "question_type": "tom:belief:inaccessible",
            "tom_type": "first-order",
            "correct_answer": "...",   # 角色真正应该持有的信念
            "wrong_answer": "...",     # 典型混淆项（更像 fact / omniscient 叙述）
            "missed_info_accessibility": "inaccessible"
        }
    """

    # 先取 short_context，没有就回退 full_context
    ctx = example.get(context_field) or example.get("full_context") or ""

    belief_qas = example.get("beliefQAs") or []
    if not belief_qas:
        raise ValueError("No beliefQAs in this example")

    # 选一个 BeliefQ[Dist]：优先带 'tom:belief' 的
    target = None
    for qa in belief_qas:
        q_type = str(qa.get("question_type", "")).lower()
        if "tom:belief" in q_type or "belief" in q_type:
            target = qa
            break
    if target is None:
        # 兜底：用第一个
        target = belief_qas[0]

    question = target.get("question", "").strip()
    belief_answer = target.get("correct_answer", "").strip()
    fact_like_answer = target.get("wrong_answer", "").strip()

    if not question:
        raise KeyError("BeliefQ entry has no 'question' field.")
    if not belief_answer:
        raise KeyError("BeliefQ entry has no 'correct_answer' field (belief gold).")

    prompt = f"""You are solving a Theory-of-Mind belief question (BeliefQ[Dist]) in the FanToM benchmark.

Conversation:
{ctx}

Question:
{question}

Answer from the character's belief perspective described in the question.
Do NOT answer from an all-knowing narrator's perspective.
Be concise and respond in one short sentence without extra explanation.
"""

    meta = {
        "set_id": example.get("set_id"),
        "part_id": example.get("part_id"),
        "conv_id": example.get("conv_id"),
        "qa_type": "BeliefQ[Dist]",
        "question_type": target.get("question_type"),
        "tom_type": target.get("tom_type"),
        "missed_info_accessibility": target.get("missed_info_accessibility"),
        # 顺便把对比用的 fact-like 答案也带回去
        "wrong_answer": fact_like_answer,
    }

    return prompt, belief_answer, meta


# answerability

def build_answerability_list_prompt_and_gold(
    example: dict,
    context_field: str = "short_context",
):
    """
    Answerability[list]：问“谁知道 precise correct answer”，输出一串角色名。
    FANTOM 里 answerabilityQA_list 是一个 dict，不是 list。
    """

    # 上层统一用的取 context 函数，你也可以直接 example.get(...)
    ctx = example.get(context_field) or example.get("full_context") or ""
    qa_block = example.get("answerabilityQA_list")

    if not qa_block:
        raise ValueError("No answerabilityQA_list in this example")
    if not isinstance(qa_block, dict):
        raise TypeError(f"answerabilityQA_list expected dict, got {type(qa_block)}")

    question = qa_block.get("question", "").strip()
    if not question:
        raise KeyError("answerabilityQA_list has no 'question' field")

    # 这里就是 ["Javier", "Sara"] 这种 list
    gold_raw = qa_block.get("correct_answer")
    if gold_raw is None:
        raise KeyError("answerabilityQA_list has no 'correct_answer' field")

    gold = _normalize_name_list(gold_raw)  # -> "Javier, Sara"

    prompt = f"""You are answering a FANToM answerability-list question.

Conversation:
{ctx}

Question:
{question}

Return ONLY the names of all characters who know the precise correct answer,
as a comma-separated list, in alphabetical order.
Do not add explanations or extra words.
"""

    meta = {
        "set_id": example.get("set_id"),
        "part_id": example.get("part_id"),
        "conv_id": example.get("conv_id"),
        "qa_type": "Answerability[List]",
    }
    return prompt, gold, meta



#infoaccess
def build_infoaccess_list_prompt_and_gold(
    example: dict,
    context_field: str = "short_context",
):
    """
    Info-Access[list]：给一个 info（隐含在对话里），问“谁知道这个 info”，输出一串角色名。
    FANTOM 里 infoAccessibilityQA_list 也是一个 dict。
    """

    ctx = example.get(context_field) or example.get("full_context") or ""
    qa_block = example.get("infoAccessibilityQA_list")

    if not qa_block:
        raise ValueError("No infoAccessibilityQA_list in this example")
    if not isinstance(qa_block, dict):
        raise TypeError(f"infoAccessibilityQA_list expected dict, got {type(qa_block)}")

    # 这一类题的 question 本身就包含了 “List all the characters who know this information.”
    question = qa_block.get("question", "").strip()
    if not question:
        raise KeyError("infoAccessibilityQA_list has no 'question' field")

    gold_raw = qa_block.get("correct_answer")
    if gold_raw is None:
        raise KeyError("infoAccessibilityQA_list has no 'correct_answer' field")

    gold = _normalize_name_list(gold_raw)  # -> "Javier, Sara"

    prompt = f"""You are answering a FANToM information-accessibility list question.

Conversation:
{ctx}

Question:
{question}

Return ONLY the names of all characters who know the relevant information,
as a comma-separated list, in alphabetical order.
Do not add explanations or extra words.
"""

    meta = {
        "set_id": example.get("set_id"),
        "part_id": example.get("part_id"),
        "conv_id": example.get("conv_id"),
        "qa_type": "Info-Access[List]",
    }
    return prompt, gold, meta




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

    # 如果 task_type 是 "All"，则测试所有类型
    if task_type == "All":
        task_types = ["FactQ", "BeliefQ", "Answerability", "InfoAccess"]
        print(f"[FANTOM] Running all task types: {task_types}")
    else:
        task_types = [task_type]
    
    # 为每种 task_type 准备 prompt 构建函数
    task_builders = {
        "FactQ": build_factq_prompt_and_gold,
        "BeliefQ": build_belief_dist_prompt_and_gold,
        "Answerability": build_answerability_list_prompt_and_gold,
        "InfoAccess": build_infoaccess_list_prompt_and_gold,
    }

    results = []
    stats = {t: {"num_correct": 0, "total_f1": 0.0, "total_q": 0} for t in task_types}

    for idx, ex in enumerate(examples, start=1):
        # 对当前 example，尝试所有 task_type
        for current_task in task_types:
            build_prompt_fn = task_builders.get(current_task)
            if not build_prompt_fn:
                continue
            
            try:
                prompt, gold, meta = build_prompt_fn(
                    ex,
                    context_field=context_field,
                )
            except Exception as e:
                # 某些 example 可能没有某种类型的 QA，跳过
                print(f"[SKIP] example {idx}, task {current_task}: {type(e).__name__}: {e}")
                continue

            pred = call_model(client, model_name, prompt)

            stats[current_task]["total_q"] += 1

            if gold:
                f1 = token_f1(pred, gold)
            else:
                f1 = 0.0

            stats[current_task]["total_f1"] += f1

            # 可选：用 F1 阈值当成一个粗略正确/错误指标
            is_correct = (f1 >= F1_THRESHOLD) if gold else False

            if is_correct:
                stats[current_task]["num_correct"] += 1

            row = {
                "global_index": idx,
                "question_index": stats[current_task]["total_q"],
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
            total_qs = sum(s["total_q"] for s in stats.values())
            print(f"[FANTOM] Processed {idx} / {len(examples)} examples "
                  f"({total_qs} total questions)")

    # 打印每种类型的统计
    
    ex = examples[0]
    print(ex["beliefQAs"][0].keys())
    print(ex["beliefQAs"][0])

    print("\n" + "=" * 80)
    for task in task_types:
        s = stats[task]
        if s["total_q"] > 0:
            mean_f1 = s["total_f1"] / s["total_q"]
            acc = s["num_correct"] / s["total_q"]
            print(f"[{task}] Questions: {s['total_q']}, "
                  f"Mean F1: {mean_f1:.4f}, "
                  f"Accuracy (F1>={F1_THRESHOLD}): {acc:.4f} ({s['num_correct']}/{s['total_q']})")
        else:
            print(f"[{task}] No valid questions processed.")
    print("=" * 80)

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
