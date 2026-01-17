from __future__ import annotations

import random
import secrets
import string
import time
from typing import Dict, List, Optional

from flask import Flask, jsonify, redirect, render_template, request, url_for

from nlp_engine import ClueEngine, ClueEngineError

app = Flask(__name__)
TURN_SECONDS = 60

games: Dict[str, dict] = {}
_NLP_ENGINE: Optional[ClueEngine] = None


def get_engine() -> ClueEngine:
    global _NLP_ENGINE
    if _NLP_ENGINE is None:
        try:
            _NLP_ENGINE = ClueEngine()
        except Exception as exc:  # pragma: no cover - surface the real error
            raise ClueEngineError(
                "NLP engine failed to load. Install dependencies and restart."
            ) from exc
    return _NLP_ENGINE


def other_color(color: str) -> str:
    return "red" if color == "blue" else "blue"


def generate_code(length: int = 4) -> str:
    alphabet = string.ascii_uppercase
    while True:
        code = "".join(secrets.choice(alphabet) for _ in range(length))
        if code not in games:
            return code


def create_game(host_name: str, mode: str) -> tuple[dict, str]:
    code = generate_code()
    host_id = secrets.token_urlsafe(8)
    mode = mode if mode in ("team", "solo") else "team"
    game = {
        "code": code,
        "host_id": host_id,
        "mode": mode,
        "solo_color": None,
        "players": {},
        "started": False,
        "start_color": None,
        "grid_words": [],
        "color_grid": [],
        "revealed": [],
        "turn": None,
        "turn_started_at": None,
        "current_clue": None,
        "guesses_left": 0,
        "winner": None,
        "status": None,
        "created_at": time.time(),
    }
    game["players"][host_id] = {
        "id": host_id,
        "name": host_name,
        "team": None,
        "is_host": True,
        "joined_at": time.time(),
    }
    games[code] = game
    return game, host_id


def add_player(game: dict, name: str) -> str:
    player_id = secrets.token_urlsafe(8)
    game["players"][player_id] = {
        "id": player_id,
        "name": name,
        "team": None,
        "is_host": False,
        "joined_at": time.time(),
    }
    return player_id


def assign_teams(game: dict) -> None:
    players = list(game["players"].values())
    random.shuffle(players)
    for idx, player in enumerate(players):
        player["team"] = "red" if idx % 2 == 0 else "blue"
    if len(players) % 2 == 0:
        game["status"] = None
    else:
        game["status"] = "Odd number of players. One team has an extra player."


def build_color_grid(start_color: str) -> List[str]:
    colors = [start_color] * 7 + [other_color(start_color)] * 6 + ["black"] + [
        "white"
    ] * 11
    random.shuffle(colors)
    return colors


def start_game(game: dict) -> None:
    pre_status = game.get("status")
    mode = game.get("mode", "team")
    if mode == "team":
        assign_teams(game)
        pre_status = game.get("status") or pre_status
    start_color = random.choice(["red", "blue"])
    game["start_color"] = start_color
    engine = get_engine()
    game["grid_words"] = engine.sample_board_words(25)
    game["color_grid"] = build_color_grid(start_color)
    game["revealed"] = [False] * 25
    game["started"] = True
    if mode == "solo":
        game["solo_color"] = start_color
    set_turn(game, start_color)
    game["winner"] = None
    if pre_status:
        game["status"] = pre_status


def set_turn(game: dict, team_color: str, reason: Optional[str] = None) -> None:
    game["turn"] = team_color
    game["turn_started_at"] = time.time()
    if reason:
        game["status"] = reason
    else:
        game["status"] = None
    try:
        clue = generate_clue(game, team_color)
        game["current_clue"] = clue or {
            "word": "pass",
            "count": 0,
            "team": team_color,
        }
    except ClueEngineError as exc:
        game["current_clue"] = {"word": "pass", "count": 0, "team": team_color}
        game["status"] = str(exc)
    try:
        game["guesses_left"] = max(0, int(game["current_clue"]["count"]))
    except (KeyError, TypeError, ValueError):
        game["guesses_left"] = 0


def advance_turn_if_expired(game: dict) -> bool:
    if not game.get("started") or game.get("winner"):
        return False
    if not game.get("turn_started_at"):
        game["turn_started_at"] = time.time()
        return False
    elapsed = time.time() - game["turn_started_at"]
    if elapsed < TURN_SECONDS:
        return False
    if game.get("mode") == "solo":
        set_turn(game, game["turn"], reason="Time expired. New clue.")
    else:
        next_color = other_color(game["turn"])
        set_turn(
            game,
            next_color,
            reason=f"Time expired. {next_color.upper()} team's turn.",
        )
    return True


def remaining_counts(game: dict) -> Dict[str, int]:
    counts = {"red": 0, "blue": 0}
    for color, revealed in zip(game["color_grid"], game["revealed"]):
        if not revealed and color in counts:
            counts[color] += 1
    return counts


def generate_clue(game: dict, team_color: str) -> Optional[dict]:
    if not game.get("started") or game.get("winner"):
        return None
    team_words = []
    opp_words = []
    neutral_words = []
    black_words = []
    for word, color, revealed in zip(
        game["grid_words"], game["color_grid"], game["revealed"]
    ):
        if revealed:
            continue
        if color == team_color:
            team_words.append(word)
        elif color == "black":
            black_words.append(word)
        elif color == "white":
            neutral_words.append(word)
        elif color in ("red", "blue"):
            opp_words.append(word)

    engine = get_engine()
    clue = engine.generate_clue(
        team_words,
        opp_words,
        neutral_words,
        black_words,
        game["grid_words"],
    )
    if not clue:
        return None
    clue["team"] = team_color
    return clue


def serialize_game(game: dict, player_id: str) -> dict:
    player = game["players"].get(player_id)
    players = [
        {
            "id": pid,
            "name": p["name"],
            "team": p["team"],
            "is_host": p["is_host"],
        }
        for pid, p in game["players"].items()
    ]

    board = []
    if game.get("started"):
        for idx, (word, color, revealed) in enumerate(
            zip(game["grid_words"], game["color_grid"], game["revealed"])
        ):
            show_color = color if revealed else "hidden"
            board.append(
                {
                    "index": idx,
                    "word": word,
                    "revealed": revealed,
                    "color": show_color,
                }
            )

    targets_remaining = None
    if game.get("started"):
        counts = remaining_counts(game)
        if game.get("mode") == "solo":
            solo_color = game.get("solo_color") or game.get("turn")
            if solo_color:
                targets_remaining = counts.get(solo_color)
    else:
        counts = {"red": None, "blue": None}

    current_clue = game["current_clue"]
    if current_clue:
        current_clue = {
            key: value
            for key, value in current_clue.items()
            if key in ("word", "count", "team")
        }

    turn_time_left = None
    if game.get("started") and game.get("turn_started_at"):
        elapsed = time.time() - game["turn_started_at"]
        turn_time_left = max(0, int(TURN_SECONDS - elapsed))

    return {
        "code": game["code"],
        "mode": game.get("mode", "team"),
        "started": game["started"],
        "start_color": game["start_color"],
        "turn": game["turn"],
        "current_clue": current_clue,
        "turn_time_left": turn_time_left,
        "guesses_left": game.get("guesses_left"),
        "targets_remaining": targets_remaining,
        "winner": game["winner"],
        "status": game["status"],
        "players": players,
        "player": player,
        "board": board,
        "remaining": counts,
    }


def request_data() -> dict:
    if request.is_json:
        return request.get_json(silent=True) or {}
    return request.form.to_dict()


@app.get("/")
def index() -> str:
    message = request.args.get("message")
    return render_template("index.html", message=message)


@app.post("/create")
def create() -> str:
    name = request.form.get("name", "").strip()
    mode = request.form.get("mode", "team").strip().lower()
    if not name:
        return redirect(url_for("index", message="Name is required."))
    game, host_id = create_game(name, mode)
    return redirect(url_for("game_view", code=game["code"], player_id=host_id))


@app.post("/join")
def join() -> str:
    name = request.form.get("name", "").strip()
    code = request.form.get("code", "").strip().upper()
    if not name or not code:
        return redirect(url_for("index", message="Name and code are required."))
    game = games.get(code)
    if not game:
        return redirect(url_for("index", message="Game code not found."))
    if game.get("mode") == "solo":
        return redirect(url_for("index", message="Solo games do not accept joins."))
    if game.get("started"):
        return redirect(url_for("index", message="Game already started."))
    player_id = add_player(game, name)
    return redirect(url_for("game_view", code=code, player_id=player_id))


@app.get("/game/<code>")
def game_view(code: str) -> str:
    game = games.get(code)
    if not game:
        return redirect(url_for("index", message="Game code not found."))
    player_id = request.args.get("player_id", "")
    player = game["players"].get(player_id)
    return render_template("game.html", game=game, player=player, player_id=player_id)


@app.get("/state/<code>")
def game_state(code: str):
    game = games.get(code)
    if not game:
        return jsonify({"error": "Game not found"}), 404
    advance_turn_if_expired(game)
    player_id = request.args.get("player_id", "")
    return jsonify(serialize_game(game, player_id))


@app.post("/start/<code>")
def start(code: str):
    game = games.get(code)
    if not game:
        return jsonify({"error": "Game not found"}), 404
    data = request_data()
    player_id = data.get("player_id", "")
    player = game["players"].get(player_id)
    if not player or not player.get("is_host"):
        return jsonify({"error": "Only the host can start the game."}), 403
    min_players = 1 if game.get("mode") == "solo" else 2
    if len(game["players"]) < min_players:
        game["status"] = (
            "Need at least 1 player to start."
            if min_players == 1
            else "Need at least 2 players to start."
        )
        return jsonify(serialize_game(game, player_id))
    try:
        start_game(game)
    except ClueEngineError as exc:
        game["status"] = str(exc)
        return jsonify({"error": str(exc)}), 500
    return jsonify(serialize_game(game, player_id))


@app.post("/clue/<code>")
def clue(code: str):
    game = games.get(code)
    if not game:
        return jsonify({"error": "Game not found"}), 404
    advance_turn_if_expired(game)
    data = request_data()
    player_id = data.get("player_id", "")
    player = game["players"].get(player_id)
    if not player:
        return jsonify({"error": "Player not found."}), 403
    if not game.get("started"):
        return jsonify({"error": "Game not started."}), 400
    if game.get("winner"):
        return jsonify({"error": "Game is over."}), 400

    return jsonify(serialize_game(game, player_id))


@app.post("/reveal/<code>/<int:index>")
def reveal(code: str, index: int):
    game = games.get(code)
    if not game:
        return jsonify({"error": "Game not found"}), 404
    advance_turn_if_expired(game)
    data = request_data()
    player_id = data.get("player_id", "")
    player = game["players"].get(player_id)
    if not player:
        return jsonify({"error": "Player not found."}), 403
    if not game.get("started"):
        return jsonify({"error": "Game not started."}), 400
    if game.get("winner"):
        return jsonify({"error": "Game is over."}), 400
    if game.get("guesses_left", 0) <= 0:
        return jsonify({"error": "No guesses left. End the turn."}), 400
    if index < 0 or index >= 25:
        return jsonify({"error": "Invalid index."}), 400
    if game["revealed"][index]:
        return jsonify(serialize_game(game, player_id))

    if game.get("mode") != "solo" and player.get("team") != game.get("turn"):
        return jsonify({"error": "Not your team's turn."}), 403

    color = game["color_grid"][index]
    game["revealed"][index] = True

    if game.get("mode") == "solo":
        if color == "black":
            game["winner"] = other_color(game["turn"])
        elif color != game["turn"]:
            set_turn(game, game["turn"], reason="Incorrect. New clue.")
        else:
            game["guesses_left"] = max(0, game.get("guesses_left", 0) - 1)
            counts = remaining_counts(game)
            if counts[game["turn"]] == 0:
                game["winner"] = game["turn"]
            elif game["guesses_left"] <= 0:
                set_turn(game, game["turn"], reason="Clue complete. New clue.")
    else:
        if color == "black":
            game["winner"] = other_color(game["turn"])
        elif color == "white":
            set_turn(game, other_color(game["turn"]))
        elif color != game["turn"]:
            set_turn(game, other_color(game["turn"]))
        else:
            game["guesses_left"] = max(0, game.get("guesses_left", 0) - 1)
            counts = remaining_counts(game)
            if counts[game["turn"]] == 0:
                game["winner"] = game["turn"]
            elif game["guesses_left"] <= 0:
                next_color = other_color(game["turn"])
                set_turn(
                    game,
                    next_color,
                    reason=f"Clue complete. {next_color.upper()} team's turn.",
                )

    return jsonify(serialize_game(game, player_id))


@app.post("/end_turn/<code>")
def end_turn(code: str):
    game = games.get(code)
    if not game:
        return jsonify({"error": "Game not found"}), 404
    expired = advance_turn_if_expired(game)
    data = request_data()
    player_id = data.get("player_id", "")
    player = game["players"].get(player_id)
    if not player:
        return jsonify({"error": "Player not found."}), 403
    if not game.get("started"):
        return jsonify({"error": "Game not started."}), 400
    if game.get("winner"):
        return jsonify({"error": "Game is over."}), 400

    if expired:
        return jsonify(serialize_game(game, player_id))

    if game.get("mode") == "solo":
        set_turn(game, game["turn"], reason="New clue.")
    else:
        set_turn(game, other_color(game["turn"]))
    return jsonify(serialize_game(game, player_id))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
