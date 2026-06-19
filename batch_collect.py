#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from catanatron import Game, Player
from catanatron.models.player import Color

from bench import (
    BASE_URL,
    STRATEGIES,
    action_to_dict,
    compact_state,
    extract_choice,
    heuristic_choice,
    hf_token_source,
    jsonable,
    parse_cutoff,
    utc_ms,
    write_jsonl,
)


DEFAULT_MODEL = "zai-org/GLM-5.2:fireworks-ai"
STRATEGY_ORDER = ["expansion-first", "city-dev-first", "balanced", "road-pressure"]


def before_cutoff(cutoff: datetime | None) -> bool:
    if cutoff is None:
        return True
    return datetime.now(cutoff.tzinfo) < cutoff


def colors() -> list[Color]:
    return [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


def random_rollout_state(seed: int, max_rollout_actions: int) -> tuple[Game, Player, list[Any]]:
    rng = random.Random(seed)
    players = [Player(color) for color in colors()]
    game = Game(players, seed=seed, vps_to_win=10)

    steps = rng.randint(0, max_rollout_actions)
    for _ in range(steps):
        if game.winning_color() is not None:
            break
        actions = list(game.state.playable_actions)
        if not actions:
            break
        action = rng.choice(actions)
        game.execute(action)

    player = game.state.current_player()
    actions = list(game.state.playable_actions)
    return game, player, actions


def make_state_record(batch: int, state_index: int, seed: int, max_rollout_actions: int) -> dict[str, Any]:
    game, player, actions = random_rollout_state(seed, max_rollout_actions)
    state_obj = compact_state(game, player.color)
    action_rows = [action_to_dict(action, i) for i, action in enumerate(actions)]
    state_id_src = f"{batch}:{state_index}:{seed}:{game.id}:{game.state.num_turns}:{player.color.value}"
    return {
        "event": "sampled_state",
        "batch": batch,
        "state_index": state_index,
        "state_id": hashlib.sha1(state_id_src.encode("utf-8")).hexdigest()[:16],
        "seed": seed,
        "game_id": game.id,
        "turn": game.state.num_turns,
        "color": player.color.value,
        "state": state_obj,
        "actions": action_rows,
    }


def prompt_from_state_record(record: dict[str, Any], strategy: str) -> str:
    return (
        "You are playing a text-only Catan benchmark through a simulator. "
        "Do not cite external rules text. Choose exactly one legal action by index.\n\n"
        f"Strategy: {strategy}. {STRATEGIES[strategy]}\n\n"
        "Return only compact JSON with this shape: "
        "{\"choice\": <integer action index>, \"reason\": \"short reason\"}.\n\n"
        "State JSON:\n"
        f"{json.dumps(record['state'], ensure_ascii=True, sort_keys=True)}\n\n"
        "Legal actions JSON:\n"
        f"{json.dumps(record['actions'], ensure_ascii=True, sort_keys=True)}"
    )


def status_from_error_text(text: str) -> str:
    lowered = text.lower()
    if "rate" in lowered or "429" in lowered:
        return "rate_limited"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    return "error"


def call_model(
    token: str,
    cutoff: datetime | None,
    model: str,
    prompt: str,
    timeout: float,
    max_tokens: int,
) -> tuple[str, int | None, str | None]:
    if not before_cutoff(cutoff):
        return "", None, "cutoff"
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You return only valid JSON for legal game-action selection.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        return "", response.status_code, response.text[:1000]
    data = response.json()
    return data["choices"][0]["message"].get("content") or "", response.status_code, None


def decision_job(
    token: str,
    cutoff: datetime | None,
    model: str,
    record: dict[str, Any],
    strategy: str,
    timeout: float,
    max_tokens: int,
) -> dict[str, Any]:
    started = utc_ms()
    prompt = prompt_from_state_record(record, strategy)
    row_id = f"{record['state_id']}:{strategy}"
    base = {
        "event": "decision_result",
        "ts_ms": utc_ms(),
        "row_id": row_id,
        "batch": record["batch"],
        "state_id": record["state_id"],
        "state_index": record["state_index"],
        "strategy": strategy,
        "model": model,
        "color": record["color"],
        "turn": record["turn"],
        "actions": record["actions"],
        "prompt": prompt,
    }
    try:
        content, http_status, error = call_model(token, cutoff, model, prompt, timeout, max_tokens)
        latency_ms = utc_ms() - started
        if error:
            fallback_choice, fallback_reason = heuristic_choice(strategy, actions_from_rows(record["actions"]))
            return {
                **base,
                "status": status_from_error_text(error) if error != "cutoff" else "cutoff",
                "http_status": http_status,
                "latency_ms": latency_ms,
                "response": None,
                "parsed": None,
                "selected_choice": fallback_choice,
                "selected_action": record["actions"][fallback_choice],
                "fallback": True,
                "fallback_reason": fallback_reason,
                "error": error,
            }
        choice, parsed = extract_choice(content, len(record["actions"]))
        if choice is None:
            fallback_choice, fallback_reason = heuristic_choice(strategy, actions_from_rows(record["actions"]))
            return {
                **base,
                "status": "fallback",
                "http_status": http_status,
                "latency_ms": latency_ms,
                "response": content,
                "parsed": parsed,
                "selected_choice": fallback_choice,
                "selected_action": record["actions"][fallback_choice],
                "fallback": True,
                "fallback_reason": "invalid_or_unparseable_model_choice",
                "error": None,
            }
        return {
            **base,
            "status": "parsed",
            "http_status": http_status,
            "latency_ms": latency_ms,
            "response": content,
            "parsed": parsed,
            "selected_choice": choice,
            "selected_action": record["actions"][choice],
            "fallback": False,
            "fallback_reason": None,
            "error": None,
        }
    except requests.Timeout as exc:
        fallback_choice, fallback_reason = heuristic_choice(strategy, actions_from_rows(record["actions"]))
        return {
            **base,
            "status": "timeout",
            "http_status": None,
            "latency_ms": utc_ms() - started,
            "response": None,
            "parsed": None,
            "selected_choice": fallback_choice,
            "selected_action": record["actions"][fallback_choice],
            "fallback": True,
            "fallback_reason": fallback_reason,
            "error": type(exc).__name__,
        }
    except Exception as exc:
        fallback_choice, fallback_reason = heuristic_choice(strategy, actions_from_rows(record["actions"]))
        return {
            **base,
            "status": "error",
            "http_status": None,
            "latency_ms": utc_ms() - started,
            "response": None,
            "parsed": None,
            "selected_choice": fallback_choice,
            "selected_action": record["actions"][fallback_choice],
            "fallback": True,
            "fallback_reason": fallback_reason,
            "error": f"{type(exc).__name__}: {str(exc)[:1000]}",
        }


class RowAction:
    def __init__(self, row: dict[str, Any]):
        self.action_type = RowType(row["action_type"])
        self.value = row.get("value")


class RowType:
    def __init__(self, value: str):
        self.value = value


def actions_from_rows(rows: list[dict[str, Any]]) -> list[RowAction]:
    return [RowAction(row) for row in rows]


def summarize_batch(batch: int, rows: list[dict[str, Any]], elapsed_ms: int) -> dict[str, Any]:
    statuses: dict[str, int] = {}
    http_statuses: dict[str, int] = {}
    latencies = []
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
        status = row.get("http_status")
        if status is not None:
            http_statuses[str(status)] = http_statuses.get(str(status), 0) + 1
        if row.get("latency_ms") is not None:
            latencies.append(row["latency_ms"])
    return {
        "event": "batch_summary",
        "ts_ms": utc_ms(),
        "batch": batch,
        "row_count": len(rows),
        "parsed_count": statuses.get("parsed", 0),
        "fallback_count": statuses.get("fallback", 0),
        "error_count": statuses.get("error", 0),
        "timeout_count": statuses.get("timeout", 0),
        "rate_limited_count": statuses.get("rate_limited", 0),
        "cutoff_count": statuses.get("cutoff", 0),
        "status_counts": statuses,
        "http_status_counts": http_statuses,
        "elapsed_ms": elapsed_ms,
        "latency_ms_min": min(latencies) if latencies else None,
        "latency_ms_max": max(latencies) if latencies else None,
        "latency_ms_avg": round(sum(latencies) / len(latencies), 1) if latencies else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--cutoff", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--states-per-batch", type=int, default=250)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--sleep-after-429", type=float, default=20.0)
    parser.add_argument("--batch-sleep", type=float, default=1.0)
    parser.add_argument("--max-rollout-actions", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means run until cutoff")
    args = parser.parse_args()

    cutoff = parse_cutoff(args.cutoff)
    root = Path(__file__).resolve().parent
    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else root / "runs" / f"batch-collect-{datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = run_dir / "decisions.jsonl"
    states_path = run_dir / "states.jsonl"
    summaries_path = run_dir / "batch_summaries.jsonl"
    status_path = run_dir / "status.json"
    token, token_source = hf_token_source()
    if not token:
        raise SystemExit("HF auth unavailable: no HF_TOKEN or cached Hugging Face token.")

    meta = {
        "event": "continuous_start",
        "ts_ms": utc_ms(),
        "model": args.model,
        "base_url": BASE_URL,
        "token_source": token_source,
        "cutoff": cutoff.isoformat() if cutoff else None,
        "workers": args.workers,
        "states_per_batch": args.states_per_batch,
        "strategy_count": len(STRATEGY_ORDER),
        "rows_per_batch": args.states_per_batch * len(STRATEGY_ORDER),
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
    }
    write_jsonl(summaries_path, meta)
    status_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    total_rows = 0
    batch = 0
    lock = threading.Lock()
    while before_cutoff(cutoff) and (args.max_batches <= 0 or batch < args.max_batches):
        started = utc_ms()
        states = [
            make_state_record(
                batch=batch,
                state_index=i,
                seed=args.seed + batch * 1_000_000 + i,
                max_rollout_actions=args.max_rollout_actions,
            )
            for i in range(args.states_per_batch)
        ]
        for state in states:
            write_jsonl(states_path, state)

        jobs = [(state, strategy) for state in states for strategy in STRATEGY_ORDER]
        rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    decision_job,
                    token,
                    cutoff,
                    args.model,
                    state,
                    strategy,
                    args.timeout,
                    args.max_tokens,
                )
                for state, strategy in jobs
                if before_cutoff(cutoff)
            ]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                rows.append(row)
                with lock:
                    write_jsonl(decisions_path, row)

        summary = summarize_batch(batch, rows, utc_ms() - started)
        total_rows += len(rows)
        summary["total_rows"] = total_rows
        write_jsonl(summaries_path, summary)
        status_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True), flush=True)

        if summary["rate_limited_count"] > 0 or summary["http_status_counts"].get("429", 0) > 0:
            time.sleep(args.sleep_after_429)
        else:
            time.sleep(args.batch_sleep)
        batch += 1

    done = {
        "event": "continuous_done",
        "ts_ms": utc_ms(),
        "batch": batch,
        "total_rows": total_rows,
        "reason": "cutoff" if not before_cutoff(cutoff) else "max_batches",
    }
    write_jsonl(summaries_path, done)
    status_path.write_text(json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(done, ensure_ascii=True, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
