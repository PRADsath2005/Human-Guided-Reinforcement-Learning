
import random
import csv
import statistics

# -----------------------------
# Environment Configuration
# -----------------------------
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)

ACTIONS = ["up", "down", "left", "right"]

ALPHA = 0.1
GAMMA = 0.9
EPISODES = 1000
MAX_STEPS = 50
EVALUATION_RUNS = 200


# -----------------------------
# Grid World Environment
# -----------------------------
def move(state, action):
    row, col = state

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


def valid_actions(state):
    return [
        action for action in ACTIONS
        if move(state, action) != state
    ]


# -----------------------------
# Q-Table
# -----------------------------
def new_q_table():
    return {
        (row, col): {
            action: 0.0 for action in ACTIONS
        }
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
    }


# -----------------------------
# Epsilon-Greedy Action Selection
# -----------------------------
def choose_action(q_table, state, epsilon):
    actions = valid_actions(state)

    if random.random() < epsilon:
        return random.choice(actions)

    best_value = max(
        q_table[state][action]
        for action in actions
    )

    best_actions = [
        action for action in actions
        if q_table[state][action] == best_value
    ]

    return random.choice(best_actions)


# -----------------------------
# Training
# -----------------------------
def train(human_guided=False, seed=42):
    random.seed(seed)

    q_table = new_q_table()
    successes = 0
    successful_steps = []

    for episode in range(EPISODES):
        state = START

        # Gradually reduce exploration
        epsilon = max(
            0.05,
            1.0 - episode / (EPISODES * 0.8)
        )

        for step in range(1, MAX_STEPS + 1):
            action = choose_action(
                q_table,
                state,
                epsilon
            )

            next_state = move(state, action)

            # Environment reward
            environment_reward = (
                20.0 if next_state == GOAL else -1.0
            )

            reward = environment_reward

            # Simulated feedback, not real human feedback
            if human_guided:
                old_distance = (
                    abs(state[0] - GOAL[0])
                    + abs(state[1] - GOAL[1])
                )

                new_distance = (
                    abs(next_state[0] - GOAL[0])
                    + abs(next_state[1] - GOAL[1])
                )

                if new_distance < old_distance:
                    reward += 2.0
                elif new_distance > old_distance:
                    reward -= 2.0

            # Q-Learning update
            old_q = q_table[state][action]

            if next_state == GOAL:
                future_q = 0.0
            else:
                next_actions = valid_actions(next_state)
                future_q = max(
                    q_table[next_state][a]
                    for a in next_actions
                )

            q_table[state][action] = old_q + ALPHA * (
                reward
                + GAMMA * future_q
                - old_q
            )

            state = next_state

            if state == GOAL:
                successes += 1
                successful_steps.append(step)
                break

    average_steps = (
        statistics.mean(successful_steps)
        if successful_steps else None
    )

    return q_table, successes, average_steps


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(q_table, runs=EVALUATION_RUNS):
    successes = 0
    successful_steps = []

    for _ in range(runs):
        state = START

        for step in range(1, MAX_STEPS + 1):
            actions = valid_actions(state)

            best_value = max(
                q_table[state][action]
                for action in actions
            )

            best_actions = [
                action for action in actions
                if q_table[state][action] == best_value
            ]

            action = random.choice(best_actions)
            state = move(state, action)

            if state == GOAL:
                successes += 1
                successful_steps.append(step)
                break

    success_rate = successes / runs * 100

    average_steps = (
        statistics.mean(successful_steps)
        if successful_steps else None
    )

    return success_rate, average_steps


# -----------------------------
# Display Results
# -----------------------------
def show_result(
    name,
    training_successes,
    training_average,
    evaluation_rate,
    evaluation_average
):
    print(f"\n{name}")
    print("-" * 42)

    print(
        f"Training success: "
        f"{training_successes}/{EPISODES}"
    )

    print(
        "Training average steps on successful episodes:",
        round(training_average, 2)
        if training_average is not None else "N/A"
    )

    print(
        f"Evaluation success rate: "
        f"{evaluation_rate:.1f}%"
    )

    print(
        "Evaluation average steps on successes:",
        round(evaluation_average, 2)
        if evaluation_average is not None else "N/A"
    )


# -----------------------------
# Save CSV Results
# -----------------------------
def save_results(results):
    filename = "experiment_results.csv"

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Method",
            "Training Success",
            "Training Episodes",
            "Training Success Rate (%)",
            "Training Average Steps",
            "Evaluation Success Rate (%)",
            "Evaluation Average Steps"
        ])

        for result in results:
            writer.writerow([
                result["method"],
                result["training_success"],
                EPISODES,
                round(
                    result["training_success"] / EPISODES * 100,
                    2
                ),
                round(result["training_avg"], 2)
                if result["training_avg"] is not None else "",
                round(result["evaluation_rate"], 2),
                round(result["evaluation_avg"], 2)
                if result["evaluation_avg"] is not None else ""
            ])

    print(f"\nResults saved to {filename}")


# -----------------------------
# Main Program
# -----------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("HUMAN-GUIDED Q-LEARNING EXPERIMENT")
    print("=" * 50)

    # Baseline training
    baseline_q, baseline_success, baseline_avg = train(
        human_guided=False,
        seed=42
    )

    baseline_rate, baseline_eval_avg = evaluate(
        baseline_q,
        seed if False else EVALUATION_RUNS
    )

    # Guided training using simulated feedback
    guided_q, guided_success, guided_avg = train(
        human_guided=True,
        seed=42
    )

    guided_rate, guided_eval_avg = evaluate(
        guided_q,
        EVALUATION_RUNS
    )

    print("\n=== EXPERIMENT RESULTS ===")

    show_result(
        "Baseline Q-Learning",
        baseline_success,
        baseline_avg,
        baseline_rate,
        baseline_eval_avg
    )

    show_result(
        "Simulated Feedback-Guided Q-Learning",
        guided_success,
        guided_avg,
        guided_rate,
        guided_eval_avg
    )

    results = [
        {
            "method": "Baseline Q-Learning",
            "training_success": baseline_success,
            "training_avg": baseline_avg,
            "evaluation_rate": baseline_rate,
            "evaluation_avg": baseline_eval_avg
        },
        {
            "method": "Simulated Feedback-Guided Q-Learning",
            "training_success": guided_success,
            "training_avg": guided_avg,
            "evaluation_rate": guided_rate,
            "evaluation_avg": guided_eval_avg
        }
    ]

    save_results(results)

    print(
        "\nNote: Guided feedback is simulated using "
        "distance to the goal."
    )
    print(
        "This is not a real human-feedback experiment yet."
    )
