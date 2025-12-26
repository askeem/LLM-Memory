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
        ret_str = "[RETRIEVED]\n" + "\n\n---\n\n".join(retrieved)
        # Greedy truncate retrieved list if needed
        if approx_tokens(ret_str) > remaining_budget:
            # Simple truncation of the string for safety
            ret_str = ret_str[:remaining_budget * 4] 
        content += "\n\n" + ret_str

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
    ap = argparse.ArgumentParser()
    # Restored original arguments
    ap.add_argument("--tasks", default="all_tasks.json")
    ap.add_argument("--budget", type=int, default=2800)
    ap.add_argument("--retrieval-budget", type=int, default=900)
    ap.add_argument("--working-budget", type=int, default=900)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--no-memory", action="store_true")
    # Added new report argument
    ap.add_argument("--report-only", action="store_true")
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
    stats = {"tasks_total": len(tasks), "tasks_ok": 0, "attempts": 0}
    
    working: List[str] = []
    results_index_lines: List[str] = []

    for i, task in enumerate(tasks):
        print(f"\n[Task {i+1}/{len(tasks)}] ID: {task['task_id']} | Type: {task['type']}")
        print(f"  Current Progress: {stats['tasks_ok']} Correct / {i} Completed")
        
        task_prompt = make_task_prompt(task)
        pinned_state = "Results Index:\n" + "\n".join(results_index_lines[-10:])

        retrieved_texts = []
        if mem:
            query = f"{task['task_id']} {task['type']} {task['description']}"
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
            log_entry = {
                "task_id": task["task_id"],
                "attempt": attempt,
                "prompt": messages,
                "response": resp["content"]
            }
            with open(f"logs/{task['task_id']}_attempt_{attempt}.json", "w") as f:
                json.dump(log_entry, f, indent=2)
                
            # Tool Handling & Diagnostics
            tool_error = False
            if resp["tool_calls"]:
                for tc in resp["tool_calls"]:
                    out = run_tool(tc["function"]["name"], json.loads(tc["function"]["arguments"]))
                    if "error" in out:
                        tool_error = True
                        working.append(f"Tool Error in {tc['function']['name']}: {out['error']}")
                    # (In a full implementation, you'd feed tool results back to the LLM here)

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
                working.append(f"Attempt {attempt} feedback: {json.dumps(details)}")

        # L1 Memory Update
        if mem:
            l1 = summarize_task_L1(task, final_ok, last_ans, last_details)
            mem.add("L1", l1, util=1.0 if final_ok else 0.2)
        
        working = [] # Clear feedback between different tasks

    print(f"\nFinal Accuracy: {(stats['tasks_ok']/stats['tasks_total'])*100:.2f}%")

if __name__ == "__main__":
    main()