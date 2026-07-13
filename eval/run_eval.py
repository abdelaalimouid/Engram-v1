"""Memory benchmark: Engram vs a stateless baseline.

Both agents use the exact same Qwen model (qwen3.7-plus) and system-prompt
style; the ONLY difference is the memory engine. The scenario spans three
sessions separated by simulated weeks. A qwen-flash judge grades answers
against expected facts.

Run:  python -m eval.run_eval
Writes eval/results.json and prints a comparison table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram import config, qwen_cloud  # noqa: E402
from engram.agent import EngramAgent  # noqa: E402
from engram.consolidation import sleep_cycle  # noqa: E402
from engram.store import MemoryStore  # noqa: E402
from eval.scenario import QUESTIONS, SESSIONS  # noqa: E402

JUDGE_PROMPT = """You are grading a memory test. The assistant was asked a question whose correct \
answer depends on facts from earlier conversations.

Question: {question}
Expected (ground truth): {expected}
Assistant's answer: {answer}

Return strict JSON: {{"correct": true|false, "reason": "<short>"}}
Grading rules:
- Judge ONLY the facts the question asks about; extra correct context is fine.
- The answer must state the up-to-date ground truth. Presenting a superseded fact as current \
(e.g. the old city or old project name as the answer) is incorrect.
- Explicitly acknowledging history is fine (e.g. "Project ABYSS, formerly ECHO" is correct)."""


class StatelessBaseline:
    """Same model, no memory engine. Each session starts from a blank context,
    which is exactly what a plain chatbot has across sessions."""

    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []

    def new_session(self) -> None:
        self.history = []

    def chat(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        reply = qwen_cloud.chat(
            [{"role": "system", "content": "You are a helpful assistant."}] + self.history[-12:],
            model=config.CHAT_MODEL,
            max_tokens=1024,
        )
        self.history.append({"role": "assistant", "content": reply})
        return reply


def judge(question: str, expected: str, answer: str) -> dict:
    try:
        return qwen_cloud.chat_json(
            [{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, expected=expected, answer=answer)}],
            max_tokens=150,
        )
    except Exception as exc:
        return {"correct": False, "reason": f"judge error: {exc}"}


def run_engram() -> dict:
    print("\n=== ENGRAM AGENT (persistent memory) ===")
    store = MemoryStore(":memory:")
    agent = EngramAgent(store)

    for session in SESSIONS[:2]:
        if session["gap_hours_before"]:
            store.timewarp(session["gap_hours_before"])
            print(f"  ⏩ timewarp +{session['gap_hours_before'] / 24:.0f} days")
        for turn in session["turns"]:
            result = agent.chat(session["id"], turn)
            agent.perceive_turn(result["user_episode_id"])  # synchronous for eval
            print(f"  [{session['id']}] user: {turn[:64]}...")
        report = sleep_cycle(store)
        print(f"  ☾ sleep cycle: {report['episodes_archived']} episodes → "
              f"{report['summaries_created']} summaries")

    store.timewarp(SESSIONS[2]["gap_hours_before"])
    print(f"  ⏩ timewarp +{SESSIONS[2]['gap_hours_before'] / 24:.0f} days, quiz session")

    results = []
    for item in QUESTIONS:
        response = agent.chat(SESSIONS[2]["id"], item["question"])
        verdict = judge(item["question"], item["expected"], response["reply"])
        results.append({
            "question": item["question"],
            "answer": response["reply"],
            "correct": bool(verdict.get("correct")),
            "reason": verdict.get("reason", ""),
            "memory_tokens": response["memory_tokens_used"],
            "recalled": len(response["recall_trace"]),
        })
        mark = "✓" if verdict.get("correct") else "✗"
        print(f"  {mark} {item['question'][:58]:<58} "
              f"(mem {response['memory_tokens_used']} tok, {len(response['recall_trace'])} recalled)")

    stats = store.stats()
    return {"results": results, "store_stats": stats}


def run_baseline() -> dict:
    print("\n=== STATELESS BASELINE (same model, no memory) ===")
    baseline = StatelessBaseline()
    for session in SESSIONS[:2]:
        baseline.new_session()
        for turn in session["turns"]:
            baseline.chat(turn)
            print(f"  [{session['id']}] user: {turn[:64]}...")

    baseline.new_session()  # quiz happens in a fresh session
    results = []
    for item in QUESTIONS:
        answer = baseline.chat(item["question"])
        verdict = judge(item["question"], item["expected"], answer)
        results.append({
            "question": item["question"],
            "answer": answer,
            "correct": bool(verdict.get("correct")),
            "reason": verdict.get("reason", ""),
        })
        mark = "✓" if verdict.get("correct") else "✗"
        print(f"  {mark} {item['question'][:58]}")
    return {"results": results}


def main() -> None:
    started = time.time()
    engram_report = run_engram()
    baseline_report = run_baseline()

    engram_score = sum(r["correct"] for r in engram_report["results"])
    baseline_score = sum(r["correct"] for r in baseline_report["results"])
    total = len(QUESTIONS)
    mean_memory_tokens = sum(r["memory_tokens"] for r in engram_report["results"]) / total

    print("\n" + "=" * 64)
    print(f"{'':24}{'Engram':>12}{'Baseline':>12}")
    print(f"{'Cross-session recall':<24}{engram_score:>9}/{total}{baseline_score:>9}/{total}")
    print(f"{'Accuracy':<24}{engram_score / total:>11.0%}{baseline_score / total:>12.0%}")
    print(f"{'Avg memory ctx (tok)':<24}{mean_memory_tokens:>12.0f}{'-':>12}")
    print(f"{'Belief supersessions':<24}"
          f"{engram_report['store_stats']['beliefs_superseded']:>12}{'-':>12}")
    print("=" * 64)
    print(f"Total wall time: {time.time() - started:.0f}s")

    output = {
        "engram": engram_report,
        "baseline": baseline_report,
        "summary": {
            "questions": total,
            "engram_correct": engram_score,
            "baseline_correct": baseline_score,
            "avg_memory_tokens": round(mean_memory_tokens, 1),
        },
    }
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
