import argparse
from decimal import Decimal, InvalidOperation
import json
import os
import re
import sys
import threading
import time

from tqdm import tqdm

MCQ_BENCHMARKS = [
    "hrbench-4k",
    "hrbench-8k",
    "vstar",
    "zoombench",
    "mme-realworld",
    "mme-realworld-cn",
    "mme-realworld-lite",
    "mmstar",
    "cv-bench",
]

POPE_BENCHMARKS = [
    "pope",
    "pope_adv",
    "pope_pop",
    "pope_random",
]

MMVP_BENCHMARKS = [
    "mmvp",
]

PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)


def extract_predicted_option(answer):
    if not isinstance(answer, str) or not answer:
        return ""
    text = answer.strip()
    labeled = re.search(
        r"(?i)\b(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*"
        r"[\(\[]?([A-F])(?:[\)\]]|(?=$)|[\s\.,:;])",
        text,
    )
    if labeled:
        return labeled.group(1).upper()
    match = re.match(r"^[\s\(\[]*([A-F])(?:[\)\]]|(?=$)|[\s\.,:;])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    parenthesized = re.findall(r"\(([A-F])\)", text, re.IGNORECASE)
    if parenthesized:
        return parenthesized[-1].upper()
    standalone = re.findall(r"(?<![A-Za-z])([A-F])(?![A-Za-z])", text, re.IGNORECASE)
    if standalone:
        return standalone[-1].upper()
    return ""


def extract_mcq_option(answer):
    if not isinstance(answer, str) or not answer:
        return ""
    text = answer.strip()
    pattern = r"^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))"
    match = re.match(pattern, text)
    if match:
        return match.group(1)
    return ""


def pope_extract(text):
    if not isinstance(text, str) or not text:
        return ""
    t = text.lstrip("*").strip()
    if t.lower().startswith("answer"):
        t = t.split("answer", 1)[1].lstrip(":").lstrip("*").strip()
    t_lower = t.lower()
    if t_lower.startswith("yes"):
        return "yes"
    if t_lower.startswith("no"):
        return "no"
    last = t.rstrip(".").rstrip("*").strip().split()[-1] if t.strip() else ""
    last = last.lstrip("*").rstrip("*").lower()
    if last in ("yes", "no"):
        return last
    return text.strip()


def mmvp_extract(text):
    if not isinstance(text, str) or not text:
        return ""
    t = text.strip().lower()
    match = re.search(r"\(([ab])\)", t)
    if match:
        return f"({match.group(1)})"
    match = re.search(r"\b([ab])\b", t)
    if match:
        return f"({match.group(1)})"
    return ""


def grade_mcq_option(gt, answer):
    gt_val = extract_mcq_option(gt)
    pred_val = extract_predicted_option(answer)
    if not gt_val or not pred_val:
        return None
    return gt_val == pred_val


def extract_numeric_answer(answer):
    if not isinstance(answer, str) or not answer.strip():
        return None
    text = answer.strip().replace(",", "")
    numbers = re.findall(r"(?<![\w.])-?(?:\d+(?:\.\d+)?|\.\d+)(?!\w)", text)
    if len(numbers) != 1:
        return None
    try:
        return Decimal(numbers[0])
    except InvalidOperation:
        return None


def grade_numeric_answer(gt, answer):
    gt_value = extract_numeric_answer(gt)
    if gt_value is None or not re.fullmatch(r"\s*-?(?:\d+(?:\.\d+)?|\.\d+)\s*", str(gt)):
        return None
    pred_value = extract_numeric_answer(answer)
    if pred_value is None:
        return None
    return gt_value == pred_value


def grade_deterministic(benchmark, gt, answer):
    if benchmark == "zoombench":
        numeric_result = grade_numeric_answer(gt, answer)
        if numeric_result is not None:
            return ("Yes" if numeric_result else "No"), "numeric_exact"
    if benchmark in MCQ_BENCHMARKS:
        mcq_result = grade_mcq_option(gt, answer)
        if mcq_result is not None:
            return ("Yes" if mcq_result else "No"), "mcq_option"
    return None


def extract_answer(model_answer_raw):
    if "<answer>" in model_answer_raw:
        start = model_answer_raw.find("<answer>")
        end = model_answer_raw.find("</answer>")
        if start != -1 and end != -1:
            return model_answer_raw[start + len("<answer>") : end].strip()
    if "Answer:" in model_answer_raw:
        return model_answer_raw[model_answer_raw.find("Answer:") :].strip()
    return model_answer_raw.strip()


def normalize_judge_output(text):
    if not isinstance(text, str):
        return "[JUDGE_API_ERROR]"
    text = text.strip()
    if text.startswith("[JUDGE_API_ERROR]"):
        return text
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>") :].strip()
    matches = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()
    lowered = text.lower()
    if re.search(r"\b(?:response|answer)\s+is\s+incorrect\b", lowered):
        return "No"
    if re.search(r"\b(?:response|answer)\s+is\s+correct\b", lowered):
        return "Yes"
    return text


def judge_via_api(
    prompts,
    api_base,
    api_key,
    judge_model,
    judge_max_tokens,
    parallel_workers=32,
    max_retries=10,
    enable_thinking=None,
):
    from openai import OpenAI

    thread_local = threading.local()

    def get_client():
        c = getattr(thread_local, "client", None)
        if c is None:
            c = OpenAI(api_key=api_key, base_url=api_base, timeout=600)
            thread_local.client = c
        return c

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [""] * len(prompts)

    def call_one(idx, prompt):
        client = get_client()
        for attempt in range(max_retries):
            try:
                extra_kwargs = {}
                if enable_thinking is not None:
                    extra_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": enable_thinking}
                    }
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=judge_max_tokens,
                    **extra_kwargs,
                )
                return idx, (resp.choices[0].message.content or "").strip()
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))
        return idx, "[JUDGE_API_ERROR]"

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(call_one, i, p) for i, p in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM Judge"):
            idx, text = future.result()
            results[idx] = text

    return results


def judge_via_vllm(prompts, judge_model_path, judge_max_tokens):
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(max_tokens=judge_max_tokens, temperature=0)
    llm = LLM(model=judge_model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(judge_model_path)

    chat_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_prompts.append(text)

    outputs = llm.generate(chat_prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def main():
    parser = argparse.ArgumentParser(description="LLM-based judge for benchmark evaluation")
    parser.add_argument("--benchmark", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--judge_model_path", default=None, type=str, help="Local model path for vLLM-based judging")
    parser.add_argument("--api_base", default=None, type=str, help="OpenAI-compatible API base URL for judging")
    parser.add_argument("--api_key", default="EMPTY", type=str)
    parser.add_argument("--judge_model", default=None, type=str, help="Model name for API-based judging")
    parser.add_argument("--judge_max_tokens", default=2048, type=int)
    parser.add_argument("--answer_dir", default="model_answer", type=str)
    parser.add_argument("--judge_dir", default="judge", type=str)
    parser.add_argument("--parallel_workers", default=32, type=int)
    parser.add_argument("--max_retries", default=10, type=int)
    parser.add_argument("--enable_thinking", choices=["True", "False"], default=None)
    args = parser.parse_args()

    if not args.api_base and not args.judge_model_path:
        print(
            "ERROR: Either --api_base (for API-based judging) or --judge_model_path "
            "(for local vLLM judging) must be provided.",
            file=sys.stderr,
        )
        sys.exit(1)

    answer_path = os.path.join(args.answer_dir, args.benchmark, f"{args.model}_answer.jsonl")
    save_path = os.path.join(args.judge_dir, args.benchmark, f"{args.model}_answer.jsonl")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    is_pope = args.benchmark in POPE_BENCHMARKS
    is_mmvp = args.benchmark in MMVP_BENCHMARKS

    data_list = []
    with open(answer_path, "r", encoding="utf-8") as f:
        for line in f:
            data_list.append(json.loads(line))

    try:
        from mathruler.grader import grade_answer

        has_mathruler = True
    except ImportError:
        has_mathruler = False

    to_llm_indices = []
    prompt_lists = []

    for i, item in enumerate(tqdm(data_list, desc="Rule-based grading")):
        question = item["query"].replace("<image>", "")
        model_answer_raw = item["model_answer"]
        extracted_answer = extract_answer(model_answer_raw)
        gt = item["response"]
        item["extracted_answer"] = extracted_answer

        is_correct = False

        if is_pope:
            pred = pope_extract(extracted_answer)
            if pred == gt.strip().lower():
                is_correct = True
                item["judge"] = "Yes"
                item["judge_source"] = "pope_exact"

        if not is_correct and is_mmvp:
            pred = mmvp_extract(extracted_answer)
            gt_norm = gt.strip().lower()
            if pred and pred == gt_norm:
                is_correct = True
                item["judge"] = "Yes"
                item["judge_source"] = "mmvp_option"

        deterministic_result = None
        if not is_correct:
            try:
                deterministic_result = grade_deterministic(args.benchmark, gt, extracted_answer)
            except Exception:
                deterministic_result = None

        if deterministic_result is not None:
            item["judge"], item["judge_source"] = deterministic_result
        elif is_correct:
            continue
        elif has_mathruler:
            try:
                is_correct = grade_answer(gt, extracted_answer)
            except Exception:
                is_correct = False
            if is_correct:
                item["judge"] = "Yes"
                item["judge_source"] = "mathruler"
                continue

        if "judge" not in item:
            prompt = PROMPT_TEMPLATE.format(gt=gt, response=extracted_answer, question=question)
            to_llm_indices.append(i)
            prompt_lists.append(prompt)

    if prompt_lists:
        print(f"Calling LLM judge for {len(prompt_lists)} remaining cases...")
        if args.api_base:
            judge_model_name = args.judge_model or "default"
            results = judge_via_api(
                prompt_lists,
                args.api_base,
                args.api_key,
                judge_model_name,
                args.judge_max_tokens,
                args.parallel_workers,
                args.max_retries,
                args.enable_thinking == "True" if args.enable_thinking is not None else None,
            )
        else:
            results = judge_via_vllm(prompt_lists, args.judge_model_path, args.judge_max_tokens)

        for idx_in_llm, response_text in enumerate(results):
            original_idx = to_llm_indices[idx_in_llm]
            data_list[original_idx]["judge"] = normalize_judge_output(response_text)
            data_list[original_idx]["judge_source"] = "llm"
            data_list[original_idx]["judge_model"] = (
                args.judge_model or args.judge_model_path or "default"
            )

    print(f"Total: {len(data_list)}, LLM used: {len(prompt_lists)}")

    with open(save_path, "w", encoding="utf-8") as out_file:
        json.dump(data_list, out_file, ensure_ascii=False, indent=4)
    print(f"Saved judge results to: {save_path}")


if __name__ == "__main__":
    main()
