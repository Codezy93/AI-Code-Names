(function () {
  const root = document.getElementById("game-root");
  if (!root) {
    return;
  }

  const gameCode = root.dataset.code;
  const playerId = root.dataset.playerId;
  const isHost = root.dataset.isHost === "true";

  const statusEl = document.getElementById("status");
  const boardEl = document.getElementById("board");
  const playersEl = document.getElementById("players");
  const clueEl = document.getElementById("clue");
  const turnEl = document.getElementById("turn");
  const timerEl = document.getElementById("timer");
  const guessesEl = document.getElementById("guesses");
  const targetsWrapEl = document.getElementById("targets-wrap");
  const targetsEl = document.getElementById("targets");
  const scoreRedEl = document.getElementById("score-red");
  const scoreBlueEl = document.getElementById("score-blue");
  const startBtn = document.getElementById("start-btn");
  const endBtn = document.getElementById("end-btn");
  let countdownId = null;
  let countdownRemaining = null;

  if (!playerId) {
    statusEl.textContent = "Missing player id.";
    return;
  }

  async function apiPost(path, payload) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Request failed.");
    }
    return data;
  }

  async function apiGet(path) {
    const response = await fetch(path, { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Request failed.");
    }
    return data;
  }

  function setStatus(message, isError) {
    statusEl.textContent = message || "";
    statusEl.classList.toggle("error", Boolean(isError));
  }

  function renderPlayers(players, mode) {
    playersEl.innerHTML = "";
    players
      .sort((a, b) => a.name.localeCompare(b.name))
      .forEach((player) => {
        const item = document.createElement("div");
        const teamClass = mode === "solo" ? "solo" : player.team || "neutral";
        item.className = `player ${teamClass}`;
        const badges = [];
        if (mode === "solo") {
          badges.push("solo");
        } else if (player.team) {
          badges.push(player.team);
        }
        if (player.is_host) {
          badges.push("host");
        }
        item.innerHTML = `
          <span class="player-name">${player.name}</span>
          <span class="player-badges">${badges.join(" | ") || "unassigned"}</span>
        `;
        playersEl.appendChild(item);
      });
  }

  function renderBoard(state) {
    boardEl.innerHTML = "";
    if (!state.started) {
      boardEl.innerHTML = '<div class="empty-board">Waiting to start.</div>';
      return;
    }

    const player = state.player || {};
    const guessesLeft =
      typeof state.guesses_left === "number" ? state.guesses_left : 0;
    const canGuess =
      !state.winner &&
      state.started &&
      guessesLeft > 0 &&
      (state.mode === "solo"
        ? true
        : player.team && state.turn && player.team === state.turn);

    state.board.forEach((cell) => {
      const card = document.createElement("button");
      card.className = "card";
      card.type = "button";
      card.textContent = cell.word;
      card.dataset.index = cell.index;

      const isHidden = cell.color === "hidden";
      const isPeek = !cell.revealed && !isHidden;
      card.classList.toggle("covered", isHidden);
      card.classList.toggle("revealed", cell.revealed);
      card.classList.toggle("peek", isPeek);
      if (!isHidden) {
        card.classList.add(cell.color);
      }

      if (!cell.revealed && canGuess) {
        card.addEventListener("click", async () => {
          try {
            await apiPost(`/reveal/${gameCode}/${cell.index}`, {
              player_id: playerId,
            });
            await refreshState();
          } catch (err) {
            setStatus(err.message, true);
          }
        });
      } else {
        card.disabled = true;
      }

      boardEl.appendChild(card);
    });
  }

  function render(state) {
    document.body.dataset.turn = state.turn || "";
    document.body.dataset.winner = state.winner || "";
    document.body.dataset.mode = state.mode || "team";
    renderPlayers(state.players || [], state.mode || "team");
    renderBoard(state);

    if (state.status) {
      setStatus(state.status, false);
    } else if (!state.started) {
      setStatus("Waiting for the host to start the game.", false);
    } else if (state.winner) {
      setStatus(`Game over. ${state.winner.toUpperCase()} wins.`, false);
    } else {
      if (state.mode === "solo") {
        setStatus("Solo mode. Tap words that match the AI clue.", false);
      } else {
        const teamLabel = state.turn ? state.turn.toUpperCase() : "READY";
        setStatus(`${teamLabel} team turn. Tap words to guess.`, false);
      }
    }

    const clue = state.current_clue;
    if (clue) {
      clueEl.textContent = `${clue.word} (${clue.count})`;
    } else {
      clueEl.textContent = "No clue yet";
    }

    if (state.mode === "solo") {
      turnEl.textContent = "Solo round";
    } else {
      turnEl.textContent = state.turn
        ? `Turn: ${state.turn.toUpperCase()}`
        : "Turn: -";
    }

    if (timerEl) {
      const active = state.started && !state.winner;
      startCountdown(active ? state.turn_time_left : null);
    }

    if (guessesEl) {
      const value =
        typeof state.guesses_left === "number" ? state.guesses_left : "-";
      guessesEl.textContent = value;
    }

    if (targetsWrapEl && targetsEl) {
      if (state.mode === "solo" && state.started) {
        const remaining =
          typeof state.targets_remaining === "number"
            ? state.targets_remaining
            : "-";
        targetsEl.textContent = remaining;
        targetsWrapEl.classList.remove("is-hidden");
      } else {
        targetsWrapEl.classList.add("is-hidden");
      }
    }

    scoreRedEl.textContent =
      state.remaining.red === null ? "-" : state.remaining.red;
    scoreBlueEl.textContent =
      state.remaining.blue === null ? "-" : state.remaining.blue;

    startBtn.classList.toggle("hidden", !isHost || state.started);
    endBtn.classList.toggle("hidden", !state.started || Boolean(state.winner));
  }

  startBtn.addEventListener("click", async () => {
    try {
      await apiPost(`/start/${gameCode}`, {
        player_id: playerId,
      });
      await refreshState();
    } catch (err) {
      setStatus(err.message, true);
    }
  });

  endBtn.addEventListener("click", async () => {
    try {
      await apiPost(`/end_turn/${gameCode}`, {
        player_id: playerId,
      });
      await refreshState();
    } catch (err) {
      setStatus(err.message, true);
    }
  });

  async function refreshState() {
    try {
      const state = await apiGet(
        `/state/${gameCode}?player_id=${encodeURIComponent(playerId)}`
      );
      render(state);
    } catch (err) {
      setStatus(err.message, true);
    }
  }

  function startCountdown(seconds) {
    if (countdownId) {
      clearInterval(countdownId);
      countdownId = null;
    }
    if (typeof seconds !== "number") {
      countdownRemaining = null;
      updateTimer("--:--");
      return;
    }
    countdownRemaining = Math.max(0, seconds);
    updateTimer(formatTime(countdownRemaining));
    countdownId = setInterval(() => {
      if (countdownRemaining === null) {
        return;
      }
      countdownRemaining = Math.max(0, countdownRemaining - 1);
      updateTimer(formatTime(countdownRemaining));
      if (countdownRemaining <= 0) {
        clearInterval(countdownId);
        countdownId = null;
        refreshState();
      }
    }, 1000);
  }

  function updateTimer(value) {
    if (timerEl) {
      timerEl.textContent = value;
    }
  }

  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  refreshState();
  setInterval(refreshState, 4000);
})();
