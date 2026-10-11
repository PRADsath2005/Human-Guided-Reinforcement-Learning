
import csv
import random
import statistics
import matplotlib.pyplot as plt

# ==================================================
# CONFIGURATION
# ==================================================
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)

ACTIONS = ["up", "down", "left", "right"]

ALPHA = 0.1
GAMMA = 0.9
EPISODES = 1000
MAX_STEPS = 50
EVALUATION_RUNS = 200

SEEDS = [11, 22, 33, 44, 55]

RESULTS_FILE = "comparison_results.csv"
GRAPH_FILE = "comparison_success_rates.png"


# ==================================================
# ENVIRONMENT
# ==================================================
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


def distance_to_goal(state):
    return (
        abs(state[0] - GOAL[0])
        + abs(state[1] - GOAL[1])
    )


# ==================================================
# Q-TABLE AND ACTION SELECTION
# ==================================================
def new_q_table():
    return {
        (row, col): {
            action: 0.0 for action in ACTIONS
        }
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
    }


def choose_action(q_table, state, epsilon, rng):
    actions = valid_actions(state)

    if rng.random() < epsilon:
        return rng.choice(actions)

    best_value = max(
        q_table[state][action]
        for action in actions
    )

    best_actions = [
        action for action in actions
        if abs(q_table[state][action] - best_value) < 1e-12
    ]

    return rng.choice(best_actions)


def update_q(q_table, state, action, next_state, reward):
    old_value = q_table[state][action]

    if next_state == GOAL:
        future_value = 0.0
    else:
        future_value = max(
            q_table[next_state][a]
            for a in valid_actions(next_state)
        )

    q_table[state][action] = old_value + ALPHA * (
        reward + GAMMA * future_value - old_value
    )


# ==================================================
# TRAINING
# ==================================================
def train(method, seed):
    rng = random.Random(seed)
    q_table = new_q_table()

    successful_episodes = 0
    successful_steps = []
    episode_success_history = []

    for episode in range(EPISODES):
        state = START

        epsilon = max(
            0.05,
            1.0 - episode / (EPISODES * 0.8)
        )

        reached_goal = False

        for step in range(1, MAX_STEPS + 1):
            action = choose_action(
                q_table, state, epsilon, rng
            )

            next_state = move(state, action)

            # Identical environment reward for both methods
            reward = 20.0 if next_state == GOAL else -1.0

            if method == "Simulated feedback-guided Q-Learning":
                # This is simulated guidance, NOT actual human feedback.
                old_distance = distance_to_goal(state)
                new_distance = distance_to_goal(next_state)

                if new_distance < old_distance:
                    reward += 2.0
                elif new_distance > old_distance:
                    reward -= 2.0

            update_q(
                q_table, state, action, next_state, reward
            )

            state = next_state

            if state == GOAL:
                successful_episodes += 1
                successful_steps.append(step)
                reached_goal = True
                break

        episode_success_history.append(
            1 if reached_goal else 0
        )

    average_steps = (
        statistics.mean(successful_steps)
        if successful_steps else None
    )

    return (
        q_table,
        successful_episodes / EPISODES * 100,
        average_steps,
        episode_success_history
    )


# ==================================================
# EVALUATION
# ==================================================
def evaluate(q_table, seed):
    rng = random.Random(seed)

    successes = 0
    successful_steps = []

    for _ in range(EVALUATION_RUNS):
        state = START

        for step in range(1, MAX_STEPS + 1):
            actions = valid_actions(state)

            best_value = max(
                q_table[state][action]
                for action in actions
            )

            best_actions = [
                action for action in actions
                if abs(
                    q_table[state][action] - best_value
                ) < 1e-12
            ]

            action = rng.choice(best_actions)
            state = move(state, action)

            if state == GOAL:
                successes += 1
                successful_steps.append(step)
                break

    success_rate = successes / EVALUATION_RUNS * 100

    average_steps = (
        statistics.mean(successful_steps)
        if successful_steps else None
    )

    return success_rate, average_steps


# ==================================================
# STATISTICS
# ==================================================
def mean_value(values):
    return statistics.mean(values) if values else 0.0


def std_value(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def average_learning_curve(histories, window=50):
    curves = []

    for history in histories:
        curve = []

        for end in range(window, len(history) + 1, window):
            chunk = history[end - window:end]
            curve.append(sum(chunk) / len(chunk) * 100)

        curves.append(curve)

    return [
        mean_value([curve[i] for curve in curves])
        for i in range(len(curves[0]))
    ]


# ==================================================
# EXPERIMENT RUNNER
# ==================================================
def run_experiment():
    methods = [
        "Baseline Q-Learning",
        "Simulated feedback-guided Q-Learning"
    ]

    all_results = []
    learning_curves = {}

    for method_index, method in enumerate(methods):
        print("\n" + "=" * 60)
        print(method)
        print("=" * 60)

        training_rates = []
        training_steps = []
        evaluation_rates = []
        evaluation_steps = []
        histories = []

        for seed in SEEDS:
            q_table, train_rate, train_avg, history = train(
                method, seed
            )

            eval_rate, eval_avg = evaluate(
                q_table, seed + 10000
            )

            training_rates.append(train_rate)
            evaluation_rates.append(eval_rate)
            histories.append(history)

            if train_avg is not None:
                training_steps.append(train_avg)

            if eval_avg is not None:
                evaluation_steps.append(eval_avg)

            print(
                f"Seed {seed}: "
                f"Training success={train_rate:.2f}%, "
                f"Evaluation success={eval_rate:.2f}%, "
                f"Evaluation average steps="
                f"{eval_avg if eval_avg is not None else 'N/A'}"
            )

        result = {
            "Method": method,
            "Seeds": len(SEEDS),
            "Episodes per seed": EPISODES,
            "Training success mean (%)": mean_value(training_rates),
            "Training success SD (%)": std_value(training_rates),
            "Training average steps": mean_value(training_steps),
            "Evaluation success mean (%)": mean_value(evaluation_rates),
            "Evaluation success SD (%)": std_value(evaluation_rates),
            "Evaluation average steps": mean_value(evaluation_steps)
        }

        all_results.append(result)
        learning_curves[method] = average_learning_curve(histories)

    # Save CSV
    fields = list(all_results[0].keys())

    with open(
        RESULTS_FILE, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for result in all_results:
            writer.writerow({
                key: (
                    round(value, 3)
                    if isinstance(value, (int, float))
                    else value
                )
                for key, value in result.items()
            })

    # Plot evaluation success rate
    names = [result["Method"] for result in all_results]
    means = [
        result["Evaluation success mean (%)"]
        for result in all_results
    ]
    deviations = [
        result["Evaluation success SD (%)"]
        for result in all_results
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(
        names,
        means,
        yerr=deviations,
        capsize=5
    )
    plt.ylabel("Evaluation Success Rate (%)")
    plt.title("Baseline vs Simulated Feedback-Guided Q-Learning")
    plt.ylim(0, 105)
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=300)
    plt.close()

    # Plot learning curves
    plt.figure(figsize=(9, 5))

    for method, curve in learning_curves.items():
        x_values = [
            (index + 1) * 50
            for index in range(len(curve))
        ]
        plt.plot(x_values, curve, label=method)

    plt.xlabel("Training Episode")
    plt.ylabel("Success Rate in Previous 50 Episodes (%)")
    plt.title("Learning Curve Comparison Across Seeds")
    plt.ylim(0, 105)
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_learning_curves.png", dpi=300)
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("MEAN RESULTS ACROSS INDEPENDENT SEEDS")
    print("=" * 60)

    for result in all_results:
        print(f"\n{result['Method']}")
        print(
            "Training success: "
            f"{result['Training success mean (%)']:.2f}% "
            f"(SD {result['Training success SD (%)']:.2f})"
        )
        print(
            "Evaluation success: "
            f"{result['Evaluation success mean (%)']:.2f}% "
            f"(SD {result['Evaluation success SD (%)']:.2f})"
        )
        print(
            "Evaluation average steps: "
            f"{result['Evaluation average steps']:.2f}"
        )

    print(f"\nSaved: {RESULTS_FILE}")
    print(f"Saved: {GRAPH_FILE}")
    print("Saved: comparison_learning_curves.png")
    print("\nResearch note:")
    print("The feedback-guided method uses simulated distance-based guidance.")
    print("It does not represent actual human feedback.")


if __name__ == "__main__":
    run_experiment()
