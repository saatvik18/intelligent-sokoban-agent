/**
 * Sokoban AI Studio — Frontend Game Engine and Hybrid AI Co-Pilot
 */

// ==========================================================================
// 1. Audio Synthesizer (Web Audio API)
// ==========================================================================
class SoundFX {
  constructor() {
    this.ctx = null;
    this.enabled = true;
  }

  init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) this.ctx = new AudioCtx();
    }
  }

  playMove() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(140, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(80, this.ctx.currentTime + 0.06);
    gain.gain.setValueAtTime(0.12, this.ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.01, this.ctx.currentTime + 0.06);
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start();
    osc.stop(this.ctx.currentTime + 0.06);
  }

  playPush() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(90, this.ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(45, this.ctx.currentTime + 0.12);
    gain.gain.setValueAtTime(0.2, this.ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.01, this.ctx.currentTime + 0.12);
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start();
    osc.stop(this.ctx.currentTime + 0.12);
  }

  playGoal() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(523.25, this.ctx.currentTime); // C5
    osc.frequency.setValueAtTime(659.25, this.ctx.currentTime + 0.08); // E5
    gain.gain.setValueAtTime(0.25, this.ctx.currentTime);
    gain.gain.linearRampToValueAtTime(0.01, this.ctx.currentTime + 0.25);
    osc.connect(gain);
    gain.connect(this.ctx.destination);
    osc.start();
    osc.stop(this.ctx.currentTime + 0.25);
  }

  playWin() {
    if (!this.enabled) return;
    this.init();
    if (!this.ctx) return;
    const notes = [523.25, 659.25, 783.99, 1046.5]; // C5, E5, G5, C6
    notes.forEach((freq, idx) => {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime + idx * 0.1);
      gain.gain.setValueAtTime(0.22, this.ctx.currentTime + idx * 0.1);
      gain.gain.linearRampToValueAtTime(0.01, this.ctx.currentTime + idx * 0.1 + 0.25);
      osc.connect(gain);
      gain.connect(this.ctx.destination);
      osc.start(this.ctx.currentTime + idx * 0.1);
      osc.stop(this.ctx.currentTime + idx * 0.1 + 0.25);
    });
  }
}

const sfx = new SoundFX();

// ==========================================================================
// 2. Game State & Variables
// ==========================================================================
let gameState = null;
let initialLevelState = null;
let moveHistory = [];
let stepCount = 0;
let pushCount = 0;
let hintTargetPos = null;

// Solver playback
let isSolving = false;
let solveQueue = [];
let solveTimer = null;
let solveSpeed = 1.0;
let isPaused = false;

// Action Map: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
const ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"];
const ACTION_DELTAS = {
  0: [-1, 0],
  1: [1, 0],
  2: [0, -1],
  3: [0, 1]
};

// ==========================================================================
// 3. Initialization & DOM Setup
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  loadLevel("microban_1");
});

function setupEventListeners() {
  // Level dropdown
  document.getElementById("level-select").addEventListener("change", (e) => {
    loadLevel(e.target.value);
  });

  // Random level button
  document.getElementById("btn-procedural").addEventListener("click", generateRandomLevel);

  // Sound toggle button
  document.getElementById("btn-sound").addEventListener("click", () => {
    sfx.enabled = !sfx.enabled;
    document.getElementById("sound-icon").textContent = sfx.enabled ? "🔊" : "🔇";
  });

  // Gameplay buttons
  document.getElementById("btn-undo").addEventListener("click", undoMove);
  document.getElementById("btn-restart").addEventListener("click", restartLevel);
  document.getElementById("btn-hint").addEventListener("click", requestHint);
  document.getElementById("btn-solve").addEventListener("click", startAutoSolve);

  // Playback control buttons
  document.getElementById("btn-play-pause").addEventListener("click", toggleSolvePause);
  document.getElementById("btn-step-forward").addEventListener("click", stepForwardSolve);
  document.getElementById("btn-stop-solve").addEventListener("click", stopAutoSolve);

  // Speed selector
  document.querySelectorAll(".btn-speed").forEach(btn => {
    btn.addEventListener("click", (e) => {
      document.querySelectorAll(".btn-speed").forEach(b => b.classList.remove("active"));
      e.target.classList.add("active");
      solveSpeed = parseFloat(e.target.dataset.speed);
      if (isSolving && !isPaused) {
        restartSolveInterval();
      }
    });
  });

  // Clear log
  document.getElementById("btn-clear-log").addEventListener("click", () => {
    document.getElementById("action-log").innerHTML = "";
  });

  // Modal actions
  document.getElementById("btn-modal-replay").addEventListener("click", () => {
    document.getElementById("win-modal").style.display = "none";
    restartLevel();
  });
  document.getElementById("btn-modal-next").addEventListener("click", () => {
    document.getElementById("win-modal").style.display = "none";
    loadNextLevel();
  });

  // Keyboard navigation
  window.addEventListener("keydown", handleKeyDown);

  // Touch / Virtual D-pad
  document.querySelectorAll(".dpad-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const act = parseInt(e.target.dataset.action);
      executeMove(act);
    });
  });
}

// ==========================================================================
// 4. API & Level Management
// ==========================================================================
async function loadLevel(levelName) {
  stopAutoSolve();
  hintTargetPos = null;
  logMessage(`Loading level: ${levelName}...`, "system");

  try {
    const res = await fetch(`/api/level?name=${encodeURIComponent(levelName)}`);
    if (!res.ok) throw new Error("Failed to load level");
    const data = await res.json();

    initialLevelState = JSON.parse(JSON.stringify(data));
    gameState = data;
    moveHistory = [];
    stepCount = 0;
    pushCount = 0;

    renderBoard();
    updateDashboard();
    logMessage(`Level '${levelName}' ready. Use Arrow Keys or WASD to move.`, "system");
  } catch (err) {
    logMessage(`Error loading level: ${err.message}`, "system");
  }
}

async function generateRandomLevel() {
  stopAutoSolve();
  hintTargetPos = null;
  logMessage("Synthesizing procedural Sokoban puzzle via reverse-pull physics...", "system");

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_boxes: 2, seed: Math.floor(Math.random() * 10000) })
    });
    if (!res.ok) throw new Error("Generation failed");
    const data = await res.json();

    initialLevelState = JSON.parse(JSON.stringify(data));
    gameState = data;
    moveHistory = [];
    stepCount = 0;
    pushCount = 0;

    renderBoard();
    updateDashboard();
    logMessage("Procedural level generated. Solvable path guaranteed!", "system");
  } catch (err) {
    logMessage(`Error generating level: ${err.message}`, "system");
  }
}

function loadNextLevel() {
  const select = document.getElementById("level-select");
  let nextIdx = (select.selectedIndex + 1) % select.options.length;
  select.selectedIndex = nextIdx;
  loadLevel(select.value);
}

function restartLevel() {
  stopAutoSolve();
  hintTargetPos = null;
  if (!initialLevelState) return;
  gameState = JSON.parse(JSON.stringify(initialLevelState));
  moveHistory = [];
  stepCount = 0;
  pushCount = 0;
  renderBoard();
  updateDashboard();
  logMessage("Level restarted to initial state.", "system");
}

function undoMove() {
  if (isSolving || moveHistory.length === 0) return;
  hintTargetPos = null;
  const prevState = moveHistory.pop();
  gameState = prevState;
  stepCount = Math.max(0, stepCount - 1);
  renderBoard();
  updateDashboard();
  sfx.playMove();
  logMessage("Move undone.", "system");
}

// ==========================================================================
// 5. Input Handling & Gameplay
// ==========================================================================
function handleKeyDown(e) {
  if (isSolving) {
    if (e.code === "Space") {
      e.preventDefault();
      toggleSolvePause();
    }
    return;
  }

  let action = null;
  switch (e.key) {
    case "ArrowUp":
    case "w":
    case "W":
      action = 0; break;
    case "ArrowDown":
    case "s":
    case "S":
      action = 1; break;
    case "ArrowLeft":
    case "a":
    case "A":
      action = 2; break;
    case "ArrowRight":
    case "d":
    case "D":
      action = 3; break;
    case "z":
    case "Z":
      undoMove(); return;
    case "r":
    case "R":
      restartLevel(); return;
    case "h":
    case "H":
      requestHint(); return;
    case " ":
      e.preventDefault();
      startAutoSolve(); return;
    default:
      return;
  }

  e.preventDefault();
  executeMove(action);
}

async function executeMove(action) {
  if (!gameState || gameState.is_solved) return;
  hintTargetPos = null;

  // Save for undo
  moveHistory.push(JSON.parse(JSON.stringify(gameState)));

  try {
    const res = await fetch("/api/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state: gameState, action: action })
    });
    const nextData = await res.json();

    if (nextData.is_valid) {
      stepCount++;
      if (nextData.is_push) {
        pushCount++;
        sfx.playPush();
      } else {
        sfx.playMove();
      }

      const prevBoxesOnTarget = gameState.num_boxes_on_target;
      gameState = nextData;

      if (gameState.num_boxes_on_target > prevBoxesOnTarget) {
        sfx.playGoal();
      }

      renderBoard();
      updateDashboard();
      logMessage(`Moved ${ACTION_NAMES[action]}${nextData.is_push ? " (Pushed Box)" : ""}`, "move");

      if (gameState.is_solved) {
        triggerWinCelebration();
      }
    } else {
      // Invalid move
      moveHistory.pop();
    }
  } catch (err) {
    console.error("Step error:", err);
  }
}

// ==========================================================================
// 6. AI Co-Pilot: Hints & Auto-Solve
// ==========================================================================
async function requestHint() {
  if (!gameState || gameState.is_solved) return;
  logMessage("AI Co-Pilot analyzing optimal next move...", "system");

  try {
    const res = await fetch("/api/advice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state: gameState })
    });
    const data = await res.json();

    const actName = data.action_name;
    const actIdx = data.recommended_action;
    const rule = data.rag_advice.primary_rule;

    // Calculate hint target coordinate
    const delta = ACTION_DELTAS[actIdx];
    hintTargetPos = [gameState.player[0] + delta[0], gameState.player[1] + delta[1]];
    renderBoard();

    logMessage(`💡 AI HINT: Move [${actName}]. ${rule}`, "system");
  } catch (err) {
    console.error("Hint error:", err);
  }
}

async function startAutoSolve() {
  if (isSolving || !gameState || gameState.is_solved) return;
  isSolving = true;
  isPaused = false;
  hintTargetPos = null;

  document.getElementById("solver-playback-bar").style.display = "flex";
  document.getElementById("btn-solve").style.display = "none";
  logMessage("🤖 Hybrid AI planning complete solution trajectory...", "system");

  try {
    const res = await fetch("/api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state: gameState })
    });
    const data = await res.json();

    if (!data.success || data.steps.length === 0) {
      logMessage("No solution found within expansion limit or puzzle is deadlock.", "system");
      stopAutoSolve();
      return;
    }

    logMessage(`A* Solution found (${data.solution_length} steps, ${data.expanded_nodes} nodes, ${data.elapsed_ms}ms). Executing...`, "system");
    solveQueue = data.steps;
    restartSolveInterval();
  } catch (err) {
    logMessage(`Solver error: ${err.message}`, "system");
    stopAutoSolve();
  }
}

function restartSolveInterval() {
  if (solveTimer) clearInterval(solveTimer);
  const intervalMs = Math.max(80, 500 / solveSpeed);

  solveTimer = setInterval(() => {
    if (isPaused) return;
    if (solveQueue.length === 0) {
      stopAutoSolve();
      return;
    }

    const step = solveQueue.shift();
    stepCount++;
    if (step.is_push) {
      pushCount++;
      sfx.playPush();
    } else {
      sfx.playMove();
    }

    gameState = step.state;
    renderBoard();
    updateDashboard();

    logMessage(`AI Step: [${step.action_name}] — ${step.rule}`, "move");

    if (gameState.is_solved) {
      stopAutoSolve();
      triggerWinCelebration();
    }
  }, intervalMs);
}

function toggleSolvePause() {
  isPaused = !isPaused;
  document.getElementById("btn-play-pause").textContent = isPaused ? "▶️" : "⏸️";
}

function stepForwardSolve() {
  if (solveQueue.length > 0) {
    const step = solveQueue.shift();
    stepCount++;
    if (step.is_push) pushCount++;
    gameState = step.state;
    renderBoard();
    updateDashboard();
    logMessage(`AI Step: [${step.action_name}]`, "move");
    if (gameState.is_solved) {
      stopAutoSolve();
      triggerWinCelebration();
    }
  }
}

function stopAutoSolve() {
  isSolving = false;
  isPaused = false;
  if (solveTimer) clearInterval(solveTimer);
  solveTimer = null;
  solveQueue = [];
  document.getElementById("solver-playback-bar").style.display = "none";
  document.getElementById("btn-solve").style.display = "inline-flex";
}

// ==========================================================================
// 7. Board Renderer & Dashboard Updates
// ==========================================================================
function renderBoard() {
  if (!gameState) return;
  const boardEl = document.getElementById("board");
  boardEl.innerHTML = "";

  const { height, width, walls, boxes, targets, player, static_deadlock_cells } = gameState;

  boardEl.style.gridTemplateColumns = `repeat(${width}, var(--tile-size))`;
  boardEl.style.gridTemplateRows = `repeat(${height}, var(--tile-size))`;

  const wallSet = new Set(walls.map(p => `${p[0]},${p[1]}`));
  const boxSet = new Set(boxes.map(p => `${p[0]},${p[1]}`));
  const targetSet = new Set(targets.map(p => `${p[0]},${p[1]}`));
  const deadlockSet = new Set((static_deadlock_cells || []).map(p => `${p[0]},${p[1]}`));

  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      const key = `${r},${c}`;
      const isWall = wallSet.has(key);
      const isBox = boxSet.has(key);
      const isTarget = targetSet.has(key);
      const isPlayer = player[0] === r && player[1] === c;
      const isDeadlock = deadlockSet.has(key);
      const isHint = hintTargetPos && hintTargetPos[0] === r && hintTargetPos[1] === c;

      const tileEl = document.createElement("div");
      tileEl.classList.add("tile");

      if (isWall) {
        tileEl.classList.add("tile-wall");
      } else {
        if (isTarget && isBox) {
          tileEl.classList.add("tile-box-target");
          const crate = document.createElement("div");
          crate.classList.add("crate");
          tileEl.appendChild(crate);
        } else if (isBox) {
          tileEl.classList.add("tile-box");
          const crate = document.createElement("div");
          crate.classList.add("crate");
          tileEl.appendChild(crate);
        } else if (isTarget) {
          tileEl.classList.add("tile-target");
        } else {
          tileEl.classList.add("tile-empty");
        }

        if (isPlayer) {
          const avatar = document.createElement("div");
          avatar.classList.add("player-avatar");
          tileEl.appendChild(avatar);
        }

        if (isDeadlock && !isTarget && !isWall) {
          tileEl.classList.add("deadlock-hazard");
          tileEl.title = "Deadlock Danger: Box entering here cannot reach any target!";
        }

        if (isHint) {
          tileEl.classList.add("hint-target");
        }
      }

      boardEl.appendChild(tileEl);
    }
  }
}

function updateDashboard() {
  if (!gameState) return;

  // Stats
  document.getElementById("stat-steps").textContent = stepCount;
  document.getElementById("stat-pushes").textContent = pushCount;
  document.getElementById("stat-boxes").textContent = `${gameState.num_boxes_on_target} / ${gameState.total_targets}`;
  document.getElementById("stat-heuristic").textContent = (gameState.bipartite_heuristic_cost || 0.0).toFixed(1);

  // RAG Card
  if (gameState.rag_advice) {
    document.getElementById("rag-primary-rule").textContent = gameState.rag_advice.primary_rule || "Preserve corridor clearance.";
    document.getElementById("rag-category-badge").textContent = (gameState.rag_advice.category || "Tactics").replace("_", " ").toUpperCase();
    document.getElementById("rag-doc-title").textContent = gameState.rag_advice.retrieved_documents ? gameState.rag_advice.retrieved_documents[0] : "Sokoban Corpus";
  }

  // Deadlock Safety Mask
  const mask = gameState.valid_action_mask || [true, true, true, true];
  const maskIds = ["mask-up", "mask-down", "mask-left", "mask-right"];
  maskIds.forEach((id, idx) => {
    const el = document.getElementById(id);
    const safe = mask[idx];
    el.className = `mask-item ${safe ? 'safe' : 'danger'}`;
    el.querySelector(".status").textContent = safe ? "SAFE" : "TRAP";
  });

  const deadBadge = document.getElementById("deadlock-status-badge");
  if (gameState.is_deadlocked && !gameState.is_solved) {
    deadBadge.className = "badge badge-danger";
    deadBadge.textContent = "DEADLOCK DETECTED";
  } else {
    deadBadge.className = "badge badge-success";
    deadBadge.textContent = "STATE SAFE";
  }

  // Telemetry
  const hCost = gameState.bipartite_heuristic_cost || 0;
  const progressPct = Math.max(5, Math.min(100, (1 - (hCost / 20)) * 100));
  document.getElementById("heuristic-progress").style.width = `${progressPct}%`;
}

function logMessage(msg, type = "system") {
  const stream = document.getElementById("action-log");
  const now = new Date();
  const timeStr = now.toTimeString().split(" ")[0].slice(3);

  const entry = document.createElement("div");
  entry.classList.add("log-entry", type);
  entry.innerHTML = `<span class="time">${timeStr}</span><span class="msg">${msg}</span>`;

  stream.appendChild(entry);
  stream.scrollTop = stream.scrollHeight;
}

function triggerWinCelebration() {
  sfx.playWin();
  logMessage("🎉 PUZZLE SOLVED! All targets placed successfully.", "win");

  document.getElementById("win-steps").textContent = stepCount;
  document.getElementById("win-pushes").textContent = pushCount;
  document.getElementById("win-optimality").textContent = stepCount <= 35 ? "Optimal ★★★" : "Completed ★★☆";
  document.getElementById("win-modal").style.display = "flex";
}
