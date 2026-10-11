
from flask import Flask, jsonify, render_template_string, request
import random
import csv
import os

app = Flask(__name__)

# -----------------------------
# Configuration
# -----------------------------
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)

ACTIONS = ["up", "down", "left", "right"]

ALPHA = 0.1
GAMMA = 0.9
MAX_STEPS = 50

FEEDBACK_FILE = "human_feedback_log.csv"
EPISODE_FILE = "human_episode_results.csv"

# -----------------------------
# Learning State
# -----------------------------
q_table = {}
state = START
pending_action = None
pending_next_state = None

episode = 1
steps = 0
total_feedback = 0
episode_successes = 0

episode_results = []
episode_reward = 0.0
episode_finished = False

message = "Click Next Action to begin."
history = []


# -----------------------------
# Environment
# -----------------------------
def new_q_table():
    return {
        (row, col): {
            action: 0.0 for action in ACTIONS
        }
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
    }


def move(position, action):
    row, col = position

    if action == "up":
        row -= 1
    elif action == "down":
        row += 1
    elif action == "left":
        col -= 1
    elif action == "right":
        col += 1

    row = max(0, min(GRID_SIZE - 1, row))
    col = max(0, min(GRID_SIZE - 1, col))

    return row, col


def valid_actions(position):
    return [
        action for action in ACTIONS
        if move(position, action) != position
    ]



def choose_action(position):
    actions = valid_actions(position)

    # Prefer actions that have the highest learned Q-value.
    best_value = max(
        q_table[position][action]
        for action in actions
    )

    best_actions = [
        action for action in actions
        if q_table[position][action] == best_value
    ]

    # Break ties by preferring movement toward the goal.
    min_distance = min(
        abs(move(position, action)[0] - GOAL[0])
        + abs(move(position, action)[1] - GOAL[1])
        for action in best_actions
    )

    closer_actions = [
        action for action in best_actions
        if (
            abs(move(position, action)[0] - GOAL[0])
            + abs(move(position, action)[1] - GOAL[1])
        ) == min_distance
    ]

    return random.choice(closer_actions)



# -----------------------------
# CSV Logging
# -----------------------------
def log_feedback(record):
    fields = [
        "Episode",
        "Step",
        "State",
        "Action",
        "Next State",
        "Feedback",
        "Reward"
    ]

    file_exists = (
        os.path.exists(FEEDBACK_FILE)
        and os.path.getsize(FEEDBACK_FILE) > 0
    )

    with open(
        FEEDBACK_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(record)


def log_episode_result():
    fields = [
        "Episode",
        "Steps",
        "Success",
        "Total Reward"
    ]

    file_exists = (
        os.path.exists(EPISODE_FILE)
        and os.path.getsize(EPISODE_FILE) > 0
    )

    with open(
        EPISODE_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "Episode": episode,
            "Steps": steps,
            "Success": int(state == GOAL),
            "Total Reward": round(episode_reward, 2)
        })


# -----------------------------
# Episode Management
# -----------------------------
def finish_episode(success):
    global episode_successes, episode_results
    global episode_finished, message

    if episode_finished:
        return

    episode_results.append({
        "episode": episode,
        "success": bool(success),
        "steps": steps,
        "reward": round(episode_reward, 2)
    })

    if success:
        episode_successes += 1

    episode_finished = True
    log_episode_result()


def reset_episode_state():
    global state, pending_action, pending_next_state
    global steps, episode_reward, episode_finished

    state = START
    pending_action = None
    pending_next_state = None
    steps = 0
    episode_reward = 0.0
    episode_finished = False


# -----------------------------
# Dashboard Data
# -----------------------------
def get_status():
    cells = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            position = (row, col)

            if position == state:
                symbol = "🤖"
            elif position == START:
                symbol = "S"
            elif position == GOAL:
                symbol = "🏁"
            else:
                symbol = ""

            cells.append({
                "row": row,
                "col": col,
                "symbol": symbol,
                "current": position == state,
                "start": position == START,
                "goal": position == GOAL
            })

    completed = len(episode_results)
    recent_results = episode_results[-10:]

    success_rate = (
        sum(item["success"] for item in episode_results)
        / completed * 100
        if completed else 0.0
    )

    recent_success_rate = (
        sum(item["success"] for item in recent_results)
        / len(recent_results) * 100
        if recent_results else 0.0
    )

    successful_steps = [
        item["steps"]
        for item in episode_results
        if item["success"]
    ]

    average_success_steps = (
        sum(successful_steps) / len(successful_steps)
        if successful_steps else None
    )

    return {
        "grid": cells,
        "state": list(state),
        "goal": list(GOAL),
        "pending_action": pending_action,
        "pending_next_state": (
            list(pending_next_state)
            if pending_next_state is not None
            else None
        ),
        "episode": episode,
        "steps": steps,
        "max_steps": MAX_STEPS,
        "total_feedback": total_feedback,
        "episode_successes": episode_successes,
        "completed_episodes": completed,
        "success_rate": round(success_rate, 2),
        "last_10_success_rate": round(
            recent_success_rate, 2
        ),
        "average_success_steps": (
            round(average_success_steps, 2)
            if average_success_steps is not None
            else None
        ),
        "episode_reward": round(episode_reward, 2),
        "episode_finished": episode_finished,
        "message": message,
        "history": history[-10:],
        "results": episode_results[-10:]
    }


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/status")
def status():
    return jsonify(get_status())


@app.route("/next", methods=["POST"])
def next_action():
    global pending_action, pending_next_state, message

    if episode_finished:
        message = "Episode finished. Click Reset Episode."
        return jsonify(get_status())

    if pending_action is not None:
        message = (
            "Please submit Good or Bad feedback "
            "for the pending action first."
        )
        return jsonify(get_status())

    if steps >= MAX_STEPS:
        finish_episode(False)
        message = (
            "Step limit reached. Episode recorded as a failure. "
            "Click Reset Episode."
        )
        return jsonify(get_status())

    pending_action = choose_action(state)
    pending_next_state = move(state, pending_action)

    message = (
        f"Agent proposes '{pending_action}' from "
        f"{state} to {pending_next_state}. "
        "Please provide your feedback."
    )

    return jsonify(get_status())


@app.route("/feedback", methods=["POST"])
def give_feedback():
    global state, pending_action, pending_next_state
    global total_feedback, message, episode_reward, steps

    data = request.get_json(silent=True) or {}
    feedback = data.get("feedback")

    if feedback not in ("good", "bad"):
        return jsonify({
            "error": "Feedback must be good or bad."
        }), 400

    if pending_action is None or pending_next_state is None:
        return jsonify({
            "error": "Click Next Action first."
        }), 400

    old_state = state
    action = pending_action
    next_state = pending_next_state

    # User-provided reward
    reward = 1.0 if feedback == "good" else -1.0

    # Environment reward for reaching the goal
    if next_state == GOAL:
        reward += 10.0

    old_q = q_table[old_state][action]

    if next_state == GOAL:
        future_q = 0.0
    else:
        next_actions = valid_actions(next_state)
        future_q = max(
            q_table[next_state][a]
            for a in next_actions
        )

    q_table[old_state][action] = old_q + ALPHA * (
        reward + GAMMA * future_q - old_q
    )

    total_feedback += 1
    episode_reward += reward

    log_feedback({
        "Episode": episode,
        "Step": steps + 1,
        "State": str(old_state),
        "Action": action,
        "Next State": str(next_state),
        "Feedback": feedback,
        "Reward": reward
    })

    history.append(
        f"Episode {episode}: {old_state} → "
        f"{action} → {next_state} ({feedback})"
    )

    state = next_state
    pending_action = None
    pending_next_state = None

    # Count this accepted action as one step.
    
    steps += 1

    if state == GOAL:
        finish_episode(True)
        message = (
            f"Goal reached in {steps} steps! "
            "Episode result recorded. Click Reset Episode."
        )
    elif steps >= MAX_STEPS:
        finish_episode(False)
        message = (
            "Step limit reached. Episode recorded as a failure. "
            "Click Reset Episode."
        )
    else:
        message = (
            f"Feedback recorded: {feedback}. "
            "Continue with Next Action."
        )

    return jsonify(get_status())


@app.route("/reset", methods=["POST"])
def reset():
    global episode, message

    if not episode_finished:
        # Do not silently count an abandoned episode as success.
        finish_episode(False)

    episode += 1
    reset_episode_state()
    message = "New episode started. Click Next Action."

    return jsonify(get_status())


@app.route("/restart", methods=["POST"])
def restart():
    global q_table, episode, total_feedback
    global episode_successes, episode_results
    global history, message

    q_table = new_q_table()
    episode = 1
    total_feedback = 0
    episode_successes = 0
    episode_results = []
    history = []

    reset_episode_state()

    # CSV logs are preserved intentionally for auditability.
    message = (
        "Learning restarted. Q-values and dashboard statistics "
        "were reset. Existing CSV logs were preserved."
    )

    return jsonify(get_status())


# -----------------------------
# Web Interface
# -----------------------------
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Human Feedback Reinforcement Learning</title>
<style>
body {
    font-family: Arial, sans-serif;
    background: #f3f6fb;
    margin: 0;
    padding: 24px;
    color: #172033;
}
.container {
    max-width: 900px;
    margin: auto;
    background: white;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 5px 24px #1b2b4515;
}
h1 { margin-top: 0; }
.subtitle { color: #64748b; line-height: 1.5; }
.grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
    max-width: 420px;
    margin: 20px auto;
}
.cell {
    aspect-ratio: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #e9eef7;
    border: 1px solid #d5deeb;
    border-radius: 9px;
    font-size: 24px;
    font-weight: bold;
}
.start { background: #d1fae5; }
.goal { background: #bbf7d0; }
.current { background: #bfdbfe; }
.controls {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    margin: 18px 0;
}
button {
    padding: 12px 16px;
    border: 0;
    border-radius: 9px;
    font-size: 15px;
    font-weight: bold;
    cursor: pointer;
}
.next { background: #2563eb; color: white; }
.good { background: #15803d; color: white; }
.bad { background: #b91c1c; color: white; }
.reset { background: #e2e8f0; color: #172033; }
button:disabled { opacity: 0.45; cursor: not-allowed; }
.status {
    background: #f1f5f9;
    padding: 14px;
    border-radius: 10px;
    min-height: 24px;
    line-height: 1.5;
}
.stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
}
.stat {
    background: #f8fafc;
    padding: 14px;
    border-radius: 10px;
    text-align: center;
}
.stat strong {
    display: block;
    font-size: 24px;
    margin-top: 7px;
}
.panel {
    background: #f8fafc;
    padding: 14px;
    border-radius: 10px;
    margin-top: 16px;
}
table {
    width: 100%;
    border-collapse: collapse;
}
th, td {
    text-align: left;
    padding: 8px;
    border-bottom: 1px solid #e2e8f0;
    font-size: 14px;
}
@media(max-width:600px) {
    body { padding: 10px; }
    .container { padding: 16px; }
    .stats { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<div class="container">
    <h1>Human Feedback RL</h1>
    <p class="subtitle">
        Q-Learning in a 5 × 5 grid world.
        Review each proposed action and provide feedback.
    </p>

    <div class="stats">
        <div class="stat">Episode<strong id="episode">1</strong></div>
        <div class="stat">Steps<strong id="steps">0 / 50</strong></div>
        <div class="stat">Total Feedback<strong id="feedbackCount">0</strong></div>
        <div class="stat">Completed Episodes<strong id="completed">0</strong></div>
        <div class="stat">Successful Episodes<strong id="successes">0</strong></div>
        <div class="stat">Overall Success Rate<strong id="successRate">0%</strong></div>
    </div>

    <div id="grid" class="grid"></div>
    <div class="status" id="message"></div>

    <div class="controls">
        <button class="next" id="nextBtn" onclick="postAction('/next')">
            Next Action
        </button>
        <button class="good" id="goodBtn"
            onclick="sendFeedback('good')" disabled>
            Good Action
        </button>
        <button class="bad" id="badBtn"
            onclick="sendFeedback('bad')" disabled>
            Bad Action
        </button>
    </div>

    <div class="controls">
        <button class="reset" onclick="postAction('/reset')">
            Reset Episode
        </button>
        <button class="reset" onclick="restartLearning()">
            Restart Learning
        </button>
    </div>

    <div class="panel">
        <h3>Performance Statistics</h3>
        <p>Success rate over the latest 10 completed episodes:
            <strong id="recentRate">0%</strong>
        </p>
        <p>Average steps in successful episodes:
            <strong id="averageSteps">N/A</strong>
        </p>
        <p>Current episode reward:
            <strong id="episodeReward">0</strong>
        </p>
    </div>

    <div class="panel">
        <h3>Recent Feedback</h3>
        <ul id="history"></ul>
    </div>

    <div class="panel">
        <h3>Recent Episode Results</h3>
        <div style="overflow-x:auto">
            <table>
                <thead>
                    <tr>
                        <th>Episode</th>
                        <th>Success</th>
                        <th>Steps</th>
                        <th>Total Reward</th>
                    </tr>
                </thead>
                <tbody id="results"></tbody>
            </table>
        </div>
    </div>

    <p class="subtitle">
        Good = +1 reward; Bad = -1 reward; reaching the goal adds +10.
        Each episode has a maximum of 50 accepted actions.
        Dashboard statistics reset with Restart Learning, while existing
        CSV logs are preserved.
    </p>
</div>

<script>
async function postAction(url) {
    const response = await fetch(url, {method: "POST"});
    const data = await response.json();
    render(data);
}

async function sendFeedback(feedback) {
    const response = await fetch("/feedback", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({feedback})
    });

    const data = await response.json();

    if (!response.ok) {
        alert(data.error || "Unable to submit feedback.");
        return;
    }

    render(data);
}

async function restartLearning() {
    if (!confirm(
        "Reset learned Q-values and dashboard statistics? " +
        "Existing CSV logs will be preserved."
    )) {
        return;
    }

    await postAction("/restart");
}

function render(data) {
    document.getElementById("episode").textContent = data.episode;
    document.getElementById("steps").textContent =
        `${data.steps} / ${data.max_steps}`;
    document.getElementById("feedbackCount").textContent =
        data.total_feedback;
    document.getElementById("completed").textContent =
        data.completed_episodes;
    document.getElementById("successes").textContent =
        data.episode_successes;
    document.getElementById("successRate").textContent =
        `${data.success_rate}%`;
    document.getElementById("recentRate").textContent =
        `${data.last_10_success_rate}%`;
    document.getElementById("averageSteps").textContent =
        data.average_success_steps === null
            ? "N/A"
            : data.average_success_steps;
    document.getElementById("episodeReward").textContent =
        data.episode_reward;
    document.getElementById("message").textContent = data.message;

    const grid = document.getElementById("grid");
    grid.innerHTML = "";

    data.grid.forEach(cell => {
        const div = document.createElement("div");
        div.className = "cell";

        if (cell.start) div.classList.add("start");
        if (cell.goal) div.classList.add("goal");
        if (cell.current) div.classList.add("current");

        div.textContent = cell.symbol;
        grid.appendChild(div);
    });

    const pending = data.pending_action !== null;
    const terminal = data.episode_finished;

    document.getElementById("nextBtn").disabled =
        pending || terminal;
    document.getElementById("goodBtn").disabled = !pending;
    document.getElementById("badBtn").disabled = !pending;

    const history = document.getElementById("history");
    history.innerHTML = "";

    data.history.slice().reverse().forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        history.appendChild(li);
    });

    const results = document.getElementById("results");
    results.innerHTML = "";

    data.results.slice().reverse().forEach(item => {
        const tr = document.createElement("tr");

        [
            item.episode,
            item.success ? "Yes" : "No",
            item.steps,
            item.reward
        ].forEach(value => {
            const td = document.createElement("td");
            td.textContent = value;
            tr.appendChild(td);
        });

        results.appendChild(tr);
    });
}

async function loadStatus() {
    const response = await fetch("/status");
    render(await response.json());
}

loadStatus();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    q_table = new_q_table()
    app.run(debug=True)
