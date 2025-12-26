"""
Main experiment runner.
FIXED: Restored original CLI arguments, fixed prompt priority, and added diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import sys
from typing import Any, Dict, List, Optional

from llm_client import LLMClient
from memory_store import MemoryStore, approx_tokens
from tools import tool_schema, run_tool
from verifier import verify

from dotenv import load_dotenv
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

def load_tasks(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)

def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text: return None
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

def make_task_prompt(task: Dict[str, Any]) -> str:
    return (
        f"Task ID: {task['task_id']}\n"
        f"Description: {task['description']}\n"
        f"Inputs: {json.dumps(task.get('inputs', {}), ensure_ascii=False)}\n"
        f"Answer keys: {task['answer_keys']}\n"
        f"Return ONLY JSON."
    )

def pack_messages(
    system_prompt: str,
    pinned_state: str,
    retrieved: List[str],
    working: List[str],
    current: str,
    budget_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Priority: system > current_task > pinned_state > working > retrieved.
    Ensures the current task is never truncated.
    """
    msgs = [{"role": "system", "content": system_prompt}]
    
    # Priority 1: Current Task
    current_part = f"[CURRENT_TASK]\n{current.strip()}"
    # Priority 2: Pinned State (Skills/Index)
    pinned_part = f"[PINNED_STATE]\n{pinned_state.strip()}"
    
    content = current_part + "\n\n" + pinned_part
    remaining_budget = budget_tokens - approx_tokens(system_prompt) - approx_tokens(content)
    
    # Priority 3: Working Feedback
    if working and remaining_budget > 100:
        work_str = "[WORKING_CONTEXT]\n" + "\n\n".join(working)
        if approx_tokens(work_str) <= remaining_budget:
            content += "\n\n" + work_str
            remaining_budget -= approx_tokens(work_str)

    # Priority 4: Retrieved Memories
    if retrieved and remaining_budget > 100:
        content += "\n\n[RETRIEVED]"
        for item in retrieved:
            item_tokens = approx_tokens(item) + 5 # plus some overhead
            if item_tokens <= remaining_budget:
                content += "\n\n---\n\n" + item
                remaining_budget -= item_tokens
            else:
                break

    msgs.append({"role": "user", "content": content})
    return msgs

def summarize_task_L1(task: Dict[str, Any], final_ok: bool, last_attempt: Dict[str, Any], details: Dict[str, Any]) -> str:
    return json.dumps({
        "task_id": task["task_id"],
        "type": task["type"],
        "ok": final_ok,
        "answer": last_attempt,
        "verifier": details,
    }, ensure_ascii=False)

def generate_monthly_report(mem: MemoryStore, llm: LLMClient, model: str):
    print("\n" + "="*50)
    print("GENERATING MONTHLY PERFORMANCE REPORT")
    print("="*50)
    
    # Retrieve top 20 successful tasks and skill cards
    memories = mem.retrieve("performance report", k=30, level_filter=["L1", "L2"])
    
    report_prompt = f"Analyze the following agent memory and write a structured monthly report:\n\n"
    for m in memories:
        report_prompt += f"[{m.level}] {m.text}\n---\n"
    
    report_prompt += "\nFormat: Executive Summary, Success Rate, Technical Insights, and Improvements."
    
    resp = llm.chat(model=model, messages=[{"role": "user", "content": report_prompt}])
    print(resp["content"])

def main() -> None:
    os.makedirs("logs", exist_ok=True)
    ap = argparse.ArgumentParser()
    # Restored original arguments
    ap.add_argument("--tasks", default="all_tasks.json")
    ap.add_argument("--budget", type=int, default=2800)
    ap.add_argument("--retrieval-budget", type=int, default=900)
    ap.add_argument("--working-budget", type=int, default=900)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--small_test", action="store_true", default=False, help="Run a small test with 5 tasks.")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    tasks_path = os.path.join(base_dir,"tasks", args.tasks)
    
    llm = LLMClient()
    solver_model = os.environ.get("LLM_MODEL", "gpt-4o") # Or your chosen model
    print(f"Using LLM model: {solver_model}")

    mem = None
    if not args.no_memory:
        mem = MemoryStore(os.path.join(base_dir, "memory", "memory.sqlite"))

    if args.report_only and mem:
        generate_monthly_report(mem, llm, solver_model)
        return

    tasks = load_tasks(tasks_path)
    
    
    working: List[str] = []
    results_index_lines: List[str] = []
    if args.small_test:
        tasks = tasks[:5]
        print(f"--- SMALL TEST MODE: Running first 5 tasks only ---")

    stats = {"tasks_total": len(tasks), "tasks_ok": 0, "attempts": 0}

    for i, task in enumerate(tasks):
        print(f"\n[Task {i+1}/{len(tasks)}] ID: {task['task_id']} | Type: {task['type']}")
        print(f"  Current Progress: {stats['tasks_ok']} Correct / {i} Completed")
        
        task_prompt = make_task_prompt(task)

        if args.no_memory:
            pinned_state = "Results Index: [DISABLED]"
        else:
            pinned_state = "Results Index:\n" + "\n".join(results_index_lines[-10:])

        retrieved_texts = []
        if mem:
            query = f"Task type: {task['type']}. {task['description']}"
            items = mem.retrieve(query, k=args.topk, token_budget=args.retrieval_budget)
            retrieved_texts = [f"[{it.level}] {it.text}" for it in items]

        final_ok = False
        last_ans = {}
        last_details = {}

        for attempt in range(1, args.tries + 1):
            stats["attempts"] += 1
            
            messages = pack_messages(
                SYSTEM_PROMPT, pinned_state, retrieved_texts, working, task_prompt, args.budget
            )
            

            resp = llm.chat(model=solver_model, messages=messages, tools=[tool_schema()])
            content = resp["content"]

                
            # Tool Handling & Diagnostics
            # Full Tool-Dialogue Loop
            tool_steps = []
            tool_error = False
            while resp["tool_calls"]:
                # Add assistant's tool call to messages
                messages.append({"role": "assistant", "content": resp["content"], "tool_calls": resp["tool_calls"]})
                
                for tc in resp["tool_calls"]:
                    fn = tc["function"]["name"]

                    tool_args = json.loads(tc["function"]["arguments"])
                    out = run_tool(fn, tool_args)
                    
                    # Add tool result to messages
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(out)})
                    tool_steps.append({"tool": fn, "args": tool_args, "out": out})
                    
                    if "error" in out:
                        tool_error = True

                # Call the LLM again with the tool results
                resp = llm.chat(model=solver_model, messages=messages, tools=[tool_schema()])
                content = resp["content"]
                log_entry = {
                    "task_id": task["task_id"],
                    "attempt": attempt,
                    "prompt": messages,
                    "response": resp["content"]
                }
                with open(f"logs/{task['task_id']}_attempt_{attempt}.json", "w") as f:
                    json.dump(log_entry, f, indent=2)
            
            # Now parse the final content after the math is done
            ans = parse_json_object(content) or {}
            ok, details = verify(task, ans, {})
            last_ans, last_details = ans, details


            if ok:
                print(f"  Attempt {attempt}: SUCCESS")
                stats["tasks_ok"] += 1
                final_ok = True
                results_index_lines.append(f"{task['task_id']}: {json.dumps(ans)}")
                break
            else:
                fail_type = "TOOL ERROR" if tool_error else "LOGIC/MATH ERROR"
                print(f"  Attempt {attempt}: FAILED ({fail_type})")

                # REVISED SCRUBBING: Remove all numerical error details and ground truth
                clean_details = {k: v for k, v in details.items() if k != 'expected'}
                
                if 'errors' in clean_details:
                    new_errors = {}
                    for k, v in clean_details['errors'].items():
                        if isinstance(v, dict):
                            # Only keep the 'got' value to show the model what it produced
                            # Explicitly remove 'expected', 'abs_err', and 'tol'
                            new_errors[k] = {"got": v.get("got")}
                        else:
                            new_errors[k] = v
                    clean_details['errors'] = new_errors

                working.append(f"Attempt {attempt} feedback: {json.dumps(clean_details)}")

        # L1 Memory Update
        if mem:
            # SCRUB the verifier details before summarization
            scrubbed_details = {k: v for k, v in last_details.items() if k != 'expected'}
            if 'errors' in scrubbed_details:
                 for k in scrubbed_details['errors']:
                    if isinstance(scrubbed_details['errors'][k], dict):
                        scrubbed_details['errors'][k].pop('expected', None)

            # Pass the scrubbed version to the summarizer
            l1 = summarize_task_L1(task, final_ok, last_ans, scrubbed_details)
            mem.add("L1", l1, util=1.0 if final_ok else 0.2)

        working = [] # Clear feedback between different tasks


    print(f"\nFinal Accuracy: {(stats['tasks_ok']/stats['tasks_total'])*100:.2f}%")

if __name__ == "__main__":
    main()