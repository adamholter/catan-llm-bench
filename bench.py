#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from catanatron import Game, Player
from catanatron.models.enums import ActionType, DEVELOPMENT_CARDS, RESOURCES
from catanatron.models.player import Color
from catanatron.state_functions import (
    get_dev_cards_in_hand,
    get_longest_road_length,
    get_played_dev_cards,
    player_key,
    player_num_resource_cards,
)
from openai import OpenAI


DEFAULT_MODELS = [
    "zai-org/GLM-5.2:zai-org",
    "zai-org/GLM-5.2:novita",
    "zai-org/GLM-5.2:fireworks-ai",
    "zai-org/GLM-5.2:deepinfra",
    "zai-org/GLM-5.2:featherless-ai",
]

BASE_URL = "https://router.huggingface.co/v1"

STRATEGIES = {
    "expansion-first": (
        "Prefer settlements, roads, and resource diversity. Expand toward useful open "
        "building locations, keep routes alive, then buy development cards or cities."
    ),
    "city-dev-first": (
        "Prefer upgrading settlements to cities, buying and playing development cards, "
        "and using robber/dev-card tempo. Expand only when city/dev-card lines are weak."
    ),
    "balanced": (
        "Balance visible points, resources, expansion, cities, and development cards. "
        "Choose the action with the best near-term point or resource outlook."
    ),
    "road-pressure": (
        "Prefer roads, longest-road pressure, and settlements. Use trades only when "
        "they immediately unlock building."
    ),
}

ACTION_PRIORITY = {
    "expansion-first": [
        "BUILD_SETTLEMENT",
        "BUILD_ROAD",
        "ROLL",
        "MARITIME_TRADE",
        "BUILD_CITY",
        "BUY_DEVELOPMENT_CARD",
        "PLAY_YEAR_OF_PLENTY",
        "PLAY_ROAD_BUILDING",
        "PLAY_KNIGHT_CARD",
        "PLAY_MONOPOLY",
        "MOVE_ROBBER",
        "DISCARD",
        "END_TURN",
    ],
    "city-dev-first": [
        "BUILD_CITY",
        "BUY_DEVELOPMENT_CARD",
        "PLAY_KNIGHT_CARD",
        "PLAY_YEAR_OF_PLENTY",
        "PLAY_MONOPOLY",
        "PLAY_ROAD_BUILDING",
        "ROLL",
        "MARITIME_TRADE",
        "BUILD_SETTLEMENT",
        "BUILD_ROAD",
        "MOVE_ROBBER",
        "DISCARD",
        "END_TURN",
    ],
    "balanced": [
        "ROLL",
        "BUILD_CITY",
        "BUILD_SETTLEMENT",
        "BUY_DEVELOPMENT_CARD",
        "BUILD_ROAD",
        "PLAY_KNIGHT_CARD",
        "PLAY_YEAR_OF_PLENTY",
        "PLAY_MONOPOLY",
        "PLAY_ROAD_BUILDING",
        "MARITIME_TRADE",
        "MOVE_ROBBER",
        "DISCARD",
        "END_TURN",
    ],
    "road-pressure": [
        "BUILD_ROAD",
        "BUILD_SETTLEMENT",
        "ROLL",
        "MARITIME_TRADE",
        "PLAY_ROAD_BUILDING",
        "BUILD_CITY",
        "BUY_DEVELOPMENT_CARD",
        "PLAY_KNIGHT_CARD",
        "MOVE_ROBBER",
        "DISCARD",
        "END_TURN",
    ],
}


def jsonable(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(jsonable(k)): jsonable(v) for k, v in value.items()}
    return value


def write_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=True, sort_keys=True) + "\n")


def utc_ms() -> int:
    return int(time.time() * 1000)


def parse_cutoff(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def hf_token_source() -> tuple[str | None, str | None]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token, "env:HF_TOKEN"
    for path in [
        Path.home() / ".cache/huggingface/token",
        Path.home() / ".huggingface/token",
    ]:
        if path.exists():
            cached = path.read_text(encoding="utf-8", errors="ignore").strip()
            if cached:
                return cached, "cache:huggingface"
    return None, None


def color_value(color: Any) -> str:
    return color.value if hasattr(color, "value") else str(color)


def action_to_dict(action: Any, index: int | None = None) -> dict[str, Any]:
    obj = {
        "color": color_value(action.color),
        "action_type": action.action_type.value,
        "value": jsonable(action.value),
        "text": f"{action.action_type.value} {json.dumps(jsonable(action.value), ensure_ascii=True)}",
    }
    if index is not None:
        obj["index"] = index
    return obj


def player_snapshot(game: Game, color: Color) -> dict[str, Any]:
    state = game.state
    key = player_key(state, color)
    resources = {resource: state.player_state[f"{key}_{resource}_IN_HAND"] for resource in RESOURCES}
    devs = {card: state.player_state[f"{key}_{card}_IN_HAND"] for card in DEVELOPMENT_CARDS}
    return {
        "color": color.value,
        "visible_vp": state.player_state[f"{key}_VICTORY_POINTS"],
        "actual_vp": state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"],
        "resource_count": player_num_resource_cards(state, color),
        "resources": resources,
        "dev_card_count": get_dev_cards_in_hand(state, color),
        "dev_cards": devs,
        "played_dev_cards": get_played_dev_cards(state, color),
        "longest_road_length": get_longest_road_length(state, color),
        "settlements": len(state.buildings_by_color[color]["SETTLEMENT"]),
        "cities": len(state.buildings_by_color[color]["CITY"]),
        "roads": len(state.buildings_by_color[color]["ROAD"]),
        "settlements_available": state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"],
        "cities_available": state.player_state[f"{key}_CITIES_AVAILABLE"],
        "roads_available": state.player_state[f"{key}_ROADS_AVAILABLE"],
    }


def compact_state(game: Game, current_color: Color) -> dict[str, Any]:
    state = game.state
    return {
        "game_id": game.id,
        "seed": game.seed,
        "vps_to_win": game.vps_to_win,
        "num_turns": state.num_turns,
        "current_color": current_color.value,
        "turn_color": state.colors[state.current_turn_index].value,
        "phase": {
            "initial_build": state.is_initial_build_phase,
            "discarding": state.is_discarding,
            "moving_knight": state.is_moving_knight,
            "road_building": state.is_road_building,
            "free_roads_available": state.free_roads_available,
            "prompt": state.current_prompt.value if hasattr(state.current_prompt, "value") else str(state.current_prompt),
        },
        "players": [player_snapshot(game, color) for color in state.colors],
        "bank": {
            "resources": dict(zip(RESOURCES, state.resource_freqdeck)),
            "development_cards_remaining": len(state.development_listdeck),
        },
    }


def build_prompt(strategy_name: str, game: Game, player: Player, actions: list[Any]) -> str:
    state_obj = compact_state(game, player.color)
    action_rows = [action_to_dict(action, i) for i, action in enumerate(actions)]
    return (
        "You are playing a text-only Catan benchmark through a simulator. "
        "Do not cite external rules text. Choose exactly one legal action by index.\n\n"
        f"Strategy: {strategy_name}. {STRATEGIES.get(strategy_name, STRATEGIES['balanced'])}\n\n"
        "Return only compact JSON with this shape: "
        "{\"choice\": <integer action index>, \"reason\": \"short reason\"}.\n\n"
        "State JSON:\n"
        f"{json.dumps(state_obj, ensure_ascii=True, sort_keys=True)}\n\n"
        "Legal actions JSON:\n"
        f"{json.dumps(action_rows, ensure_ascii=True, sort_keys=True)}"
    )


def extract_choice(text: str, n_actions: int) -> tuple[int | None, dict[str, Any]]:
    raw = text.strip()
    candidates = [raw]
    match = re.search(r"\{.*\}", raw, re.S)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            choice = obj.get("choice", obj.get("action", obj.get("index")))
            if isinstance(choice, str) and choice.isdigit():
                choice = int(choice)
            if isinstance(choice, int) and 0 <= choice < n_actions:
                return choice, obj
        except Exception:
            pass
    number = re.search(r"\b(\d+)\b", raw)
    if number:
        choice = int(number.group(1))
        if 0 <= choice < n_actions:
            return choice, {"parsed_from_text": raw[:500]}
    return None, {"unparsed": raw[:1000]}


def heuristic_choice(strategy: str, actions: list[Any]) -> tuple[int, str]:
    priority = ACTION_PRIORITY.get(strategy, ACTION_PRIORITY["balanced"])
    for wanted in priority:
        matches = [
            i
            for i, action in enumerate(actions)
            if action.action_type.value == wanted
        ]
        if matches:
            return matches[0], f"heuristic priority {wanted}"
    return 0, "heuristic default first legal action"


@dataclass
class HFClient:
    token: str
    models: list[str]
    cutoff: datetime | None
    log_path: Path
    temperature: float = 0.2

    def __post_init__(self) -> None:
        self.client = OpenAI(base_url=BASE_URL, api_key=self.token)

    def before_cutoff(self) -> bool:
        if self.cutoff is None:
            return True
        return datetime.now(self.cutoff.tzinfo) < self.cutoff

    def choose(self, prompt: str, game_id: str, action_index: int, color: str) -> tuple[str | None, str | None]:
        if not self.before_cutoff():
            write_jsonl(
                self.log_path,
                {
                    "event": "api_cutoff",
                    "ts_ms": utc_ms(),
                    "game_id": game_id,
                    "action_index": action_index,
                    "color": color,
                    "cutoff": self.cutoff.isoformat(),
                },
            )
            return None, None

        for model in self.models:
            if not self.before_cutoff():
                return None, None
            started = utc_ms()
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You return only valid JSON for legal game-action selection.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=120,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                write_jsonl(
                    self.log_path,
                    {
                        "event": "model_response",
                        "ts_ms": utc_ms(),
                        "game_id": game_id,
                        "action_index": action_index,
                        "color": color,
                        "model": model,
                        "latency_ms": utc_ms() - started,
                        "response": content,
                    },
                )
                return content, model
            except Exception as exc:
                write_jsonl(
                    self.log_path,
                    {
                        "event": "provider_error",
                        "ts_ms": utc_ms(),
                        "game_id": game_id,
                        "action_index": action_index,
                        "color": color,
                        "model": model,
                        "latency_ms": utc_ms() - started,
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1000],
                    },
                )
        return None, None


class BenchDecider:
    def __init__(
        self,
        mode: str,
        strategies_by_color: dict[str, str],
        log_path: Path,
        hf_client: HFClient | None,
    ):
        self.mode = mode
        self.strategies_by_color = strategies_by_color
        self.log_path = log_path
        self.hf_client = hf_client
        self.action_counter = 0
        self.real_calls = 0
        self.real_actions = 0
        self.model_counts: dict[str, int] = {}
        self.fallbacks = 0
        self.action_type_counts: dict[str, int] = {}

    def __call__(self, player: Player, game: Game, playable_actions: list[Any]) -> Any:
        actions = list(playable_actions)
        color = color_value(player.color)
        strategy = self.strategies_by_color[color]
        prompt = build_prompt(strategy, game, player, actions)
        event_base = {
            "ts_ms": utc_ms(),
            "event": "decision",
            "game_id": game.id,
            "action_index": self.action_counter,
            "turn": game.state.num_turns,
            "color": color,
            "strategy": strategy,
            "mode": self.mode,
            "prompt": prompt,
            "actions": [action_to_dict(action, i) for i, action in enumerate(actions)],
        }

        choice = None
        parsed: dict[str, Any] = {}
        model = None
        response = None
        fallback_reason = None
        used_fallback = False

        if self.mode == "hf" and self.hf_client is not None:
            response, model = self.hf_client.choose(prompt, game.id, self.action_counter, color)
            if response is not None:
                self.real_calls += 1
                choice, parsed = extract_choice(response, len(actions))
                if choice is None:
                    fallback_reason = "invalid_or_unparseable_model_choice"
            else:
                fallback_reason = "no_model_response"

        if choice is None:
            choice, fallback_detail = heuristic_choice(strategy, actions)
            used_fallback = True
            self.fallbacks += 1
            fallback_reason = fallback_reason or fallback_detail
            parsed = {"fallback_detail": fallback_detail}
        else:
            self.real_actions += 1
            self.model_counts[model or "unknown"] = self.model_counts.get(model or "unknown", 0) + 1

        action = actions[choice]
        action_type = action.action_type.value
        self.action_type_counts[action_type] = self.action_type_counts.get(action_type, 0) + 1
        write_jsonl(
            self.log_path,
            {
                **event_base,
                "response": response,
                "model": model,
                "parsed": parsed,
                "selected_choice": choice,
                "selected_action": action_to_dict(action),
                "fallback": used_fallback,
                "fallback_reason": fallback_reason,
            },
        )
        self.action_counter += 1
        return action


def colors() -> list[Color]:
    return [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


def parse_strategies(value: str, n: int) -> list[str]:
    requested = [x.strip() for x in value.split(",") if x.strip()]
    if not requested:
        requested = ["expansion-first", "city-dev-first"]
    for strategy in requested:
        if strategy not in STRATEGIES:
            raise SystemExit(f"Unknown strategy {strategy}. Choose from: {', '.join(STRATEGIES)}")
    while len(requested) < n:
        requested.extend(requested[: n - len(requested)])
    return requested[:n]


def result_summary(game: Game, strategies_by_color: dict[str, str], decider: BenchDecider, seed: int) -> dict[str, Any]:
    winner = game.winning_color()
    return {
        "event": "game_result",
        "ts_ms": utc_ms(),
        "game_id": game.id,
        "seed": seed,
        "winner": color_value(winner) if winner is not None else None,
        "winner_strategy": strategies_by_color.get(color_value(winner)) if winner is not None else None,
        "num_turns": game.state.num_turns,
        "actions": len(game.state.actions),
        "decisions": decider.action_counter,
        "real_model_calls": decider.real_calls,
        "real_model_actions": decider.real_actions,
        "fallbacks": decider.fallbacks,
        "model_counts": decider.model_counts,
        "action_type_counts": decider.action_type_counts,
        "players": [player_snapshot(game, color) for color in game.state.colors],
        "strategies_by_color": strategies_by_color,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "fake", "hf"], default="auto")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=200)
    parser.add_argument("--vps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategies", default="expansion-first,city-dev-first")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--cutoff", default="")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    cutoff = parse_cutoff(args.cutoff)
    token, token_source = hf_token_source()
    mode = args.mode
    if mode == "auto":
        mode = "hf" if token else "fake"
    if mode == "hf" and not token:
        print("HF auth unavailable: no HF_TOKEN or cached Hugging Face token. Running fake mode instead.", file=sys.stderr)
        mode = "fake"

    root = Path(__file__).resolve().parent
    run_dir = Path(args.run_dir) if args.run_dir else root / "runs" / datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "events.jsonl"

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    meta = {
        "event": "run_start",
        "ts_ms": utc_ms(),
        "mode": mode,
        "requested_mode": args.mode,
        "base_url": BASE_URL if mode == "hf" else None,
        "models": models if mode == "hf" else [],
        "token_source": token_source if token else None,
        "cutoff": cutoff.isoformat() if cutoff else None,
        "games": args.games,
        "max_actions": args.max_actions,
        "vps": args.vps,
        "seed": args.seed,
    }
    write_jsonl(log_path, meta)

    hf_client = HFClient(token=token, models=models, cutoff=cutoff, log_path=log_path) if mode == "hf" and token else None
    aggregate: list[dict[str, Any]] = []
    strategy_names = parse_strategies(args.strategies, 4)

    for game_i in range(args.games):
        seed = args.seed + game_i
        players = [Player(color) for color in colors()]
        game = Game(players, seed=seed, vps_to_win=args.vps)
        strategies_by_color = {
            color_value(player.color): strategy_names[i] for i, player in enumerate(players)
        }
        write_jsonl(
            log_path,
            {
                "event": "game_start",
                "ts_ms": utc_ms(),
                "game_id": game.id,
                "game_number": game_i,
                "seed": seed,
                "seating_order": [color.value for color in game.state.colors],
                "strategies_by_color": strategies_by_color,
            },
        )
        decider = BenchDecider(mode, strategies_by_color, log_path, hf_client)
        while game.winning_color() is None and decider.action_counter < args.max_actions:
            game.play_tick(decide_fn=decider)
        summary = result_summary(game, strategies_by_color, decider, seed)
        write_jsonl(log_path, summary)
        aggregate.append(summary)
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True))

    run_summary = {
        "event": "run_result",
        "ts_ms": utc_ms(),
        "run_dir": str(run_dir),
        "mode": mode,
        "real_model_calls_made": sum(x["real_model_calls"] for x in aggregate),
        "real_model_actions_selected": sum(x["real_model_actions"] for x in aggregate),
        "total_games": len(aggregate),
        "total_turns": sum(x["num_turns"] for x in aggregate),
        "total_actions": sum(x["actions"] for x in aggregate),
        "winners": [x["winner"] for x in aggregate],
        "winner_strategies": [x["winner_strategy"] for x in aggregate],
        "log_path": str(log_path),
    }
    (run_dir / "summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(log_path, run_summary)
    print(json.dumps(run_summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
