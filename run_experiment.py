"""
Main experiment runner with Hierarchical Memory, Structured Logging, and robust Tool Execution.
Company-aware retrieval for consultant/client playbooks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from llm_client import LLMClient
from memory_store import MemoryStore
from tools import tool_schema, run_tool
from verifier import verify

load_dotenv()

SYSTEM_PROMPT = """You are a corporate-finance problem solver.
Rules:
- You MUST output a single JSON object with exactly the keys requested by `answer_keys`.
- Use the calculator tool for arithmetic when helpful.
- Assume cashflows are annual with CF0 at t=0 and CFt at t=t.
- If a task references earlier task IDs, use memory to recall their inputs/results.
- Inside the calc tool, you ONLY have access to: math, np (numpy), sum, range, and basic arithmetic. Do NOT try to import other libraries.
- The calc tool must return a single number (float/int) or a list of numbers — not a range, dict, or iterator.
Output numbers as decimals (not percentages), unless the task explicitly asks otherwise.
"""

def setup_logging_dir(base_dir: str, model_name: str, use_memory: bool) -> str:
    subtype = "with_memory" if use_memory else "no_memory"
    date_str = time.strftime("%Y%m%d")
    path = os.path.join(base_dir, subtype, date_str)
    os.makedirs(path, exist_ok=True)

    run_num = 1
    while True:
        run_name = f"{model_name}_{run_num:03d}"
        full_path = os.path.join(path, run_name)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            return full_path
        run_num += 1

def load_tasks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            return None
    return None

def generate_summary(llm: LLMClient, model: str, text: str) -> str:
    """Creates a 1-sentence summary for the compressed memory view."""
    try:
        resp = llm.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": f"Summarize this corporate-finance methodology in one sentence (max 15 words):\n{text}"
            }],
            max_output_tokens=50
        )
        s = (resp.get("content") or "").strip()
        if not s:
            return text[:120] + "..."
        return s
    except Exception:
        return text[:120] + "..."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="all_tasks.json")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--tries", type=int, default=3)
    parser.add_argument("--small_test", action="store_true", help="Run a small test with 10 tasks.")
    args = parser.parse_args()

    # Environment checks
    if not os.environ.get("LLM_MODEL"):
        print("Error: LLM_MODEL environment variable not set.")
        sys.exit(1)
    if not os.environ.get("LLM_SUMMARIZER"):
        print("Error: LLM_SUMMARIZER environment variable not set.")
        sys.exit(1)

    # Setup logging
    log_dir = setup_logging_dir("logs", os.environ.get("LLM_MODEL"), not args.no_memory)
    print(f"Logging to: {log_dir}")

    # Setup components
    llm = LLMClient()
    mem = MemoryStore(os.path.join(log_dir, "memory.sqlite")) if not args.no_memory else None

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_path = os.path.join(base_dir, "tasks", args.tasks)
    tasks = load_tasks(tasks_path)

    if args.small_test:
        tasks = tasks[:10]
        print("--- SMALL TEST MODE: Running first 10 tasks only ---")

    stats = {"ok": 0, "total": 0, "attempts": 0}

    # Working context (L0) - reset per task
    working: List[str] = []

    for i, task in enumerate(tasks):
        stats["total"] += 1
        t_id = task["task_id"]
        t_type = task["type"]

        print(f"\n[{i+1}/{len(tasks)}] Task {t_id} ({t_type})")
        print(f"  Current Progress: {stats['ok']} Correct")

        # Company (optional)
        company = None
        try:
            company = task.get("inputs", {}).get("company")
        except Exception:
            company = None

        # Retrieve memory
        mem_context = ""
        if mem:
            # Improve query by explicitly including company
            q_company = f" company={company} " if company else ""
            query = f"{q_company}{task['description']} {json.dumps(task['inputs'], ensure_ascii=False)}"
            company = task.get("inputs", {}).get("company")
            items = mem.retrieve(query, k=5, company=company, folder_hint=t_type)
            if items:
                mem_context = "\n[RELEVANT MEMORY]\n" + mem.format_memories(items)
                print(f"  Retrieved {len(items)} memories" + (f" (company={company})" if company else ""))

        final_ok = False

        for attempt in range(1, args.tries + 1):
            stats["attempts"] += 1

            # Construct prompt
            user_content = (
                f"Task: {task['description']}\n"
                f"Inputs: {task['inputs']}\n"
                f"Keys: {task['answer_keys']}\n"
            )
            if mem_context:
                user_content += f"{mem_context}\n"

            if working:
                user_content += "\n[PREVIOUS ATTEMPTS]\n" + "\n".join(working)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            # LLM chat
            resp = llm.chat(model=os.environ.get("LLM_MODEL"), messages=messages, tools=[tool_schema()])
            content = resp.get("content") or ""

            # Tool loop
            tool_steps = []
            tool_error = False

            while resp.get("tool_calls"):
                messages.append({"role": "assistant", "content": resp.get("content"), "tool_calls": resp.get("tool_calls")})

                for tc in resp["tool_calls"]:
                    fn = tc["function"]["name"]
                    try:
                        args_dict = json.loads(tc["function"]["arguments"])
                    except Exception:
                        args_dict = {}

                    out = run_tool(fn, args_dict)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(out)})
                    tool_steps.append({"tool": fn, "args": args_dict, "out": out})

                    if isinstance(out, dict) and "error" in out:
                        tool_error = True

                resp = llm.chat(model=os.environ.get("LLM_MODEL"), messages=messages, tools=[tool_schema()])
                content = resp.get("content") or ""

            # Save raw log
            log_entry = {
                "task_id": t_id,
                "attempt": attempt,
                "prompt": messages,
                "response": content
            }
            with open(os.path.join(log_dir, f"{t_id}_attempt_{attempt}.json"), "w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)

            # Verification
            ans = parse_json_object(content) or {}
            ok, details = verify(task, ans, {})

            if ok:
                print(f"  Attempt {attempt}: SUCCESS")
                stats["ok"] += 1
                final_ok = True

                # Update memory on success
                if mem:
                    mem_status = "worked"
                    meta = {
                        "company": company,
                        "task_type": t_type,
                        "task_id": t_id,
                        "answer_keys": task.get("answer_keys", []),
                    }
                    memory_text = (
                        f"TaskID: {t_id}\n"
                        f"Company: {company}\n"
                        f"Type: {t_type}\n"
                        f"Task: {task['description']}\n"
                        f"Inputs: {json.dumps(task['inputs'], ensure_ascii=False)}\n"
                        f"ModelAnswer: {content}\n"
                        f"Result: SUCCESS\n"
                    )
                    summary = generate_summary(llm, os.environ.get("LLM_SUMMARIZER"), memory_text)
                    mem.add(folder_name=t_type, text=memory_text, summary=summary, status=mem_status, meta=meta)
                    print("  Memory updated (Worked).")
                break

            else:
                fail_type = "TOOL ERROR" if tool_error else "LOGIC/MATH ERROR"
                print(f"  Attempt {attempt}: FAILED ({fail_type})")

                # scrub details for feedback loop (keep got, optionally keep expected if you want faster correction)
                clean_details = {k: v for k, v in details.items() if k != "expected"}
                if "errors" in clean_details:
                    new_errors = {}
                    for k, v in clean_details["errors"].items():
                        if isinstance(v, dict):
                            new_errors[k] = {"got": v.get("got")}
                        else:
                            new_errors[k] = v
                    clean_details["errors"] = new_errors

                working.append(f"Attempt {attempt} feedback: {json.dumps(clean_details)}")

                # Update memory on final failure
                if attempt == args.tries and mem:
                    mem_status = "failed"
                    meta = {
                        "company": company,
                        "task_type": t_type,
                        "task_id": t_id,
                        "answer_keys": task.get("answer_keys", []),
                    }
                    memory_text = (
                        f"TaskID: {t_id}\n"
                        f"Company: {company}\n"
                        f"Type: {t_type}\n"
                        f"Task: {task['description']}\n"
                        f"Inputs: {json.dumps(task['inputs'], ensure_ascii=False)}\n"
                        f"ModelAnswer: {content}\n"
                        f"Result: FAILED ({fail_type})\n"
                    )
                    summary = generate_summary(llm, os.environ.get("LLM_SUMMARIZER"), memory_text)
                    mem.add(folder_name=t_type, text=memory_text, summary=summary, status=mem_status, meta=meta)
                    print("  Memory updated (Failed).")

        working = []  # reset L0 context

    print(f"\nRun complete. Stats: {stats}")
    print(f"Accuracy: {(stats['ok']/stats['total'])*100:.2f}%")

if __name__ == "__main__":
    main()
