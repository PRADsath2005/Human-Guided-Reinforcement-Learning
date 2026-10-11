
import csv
import random
import statistics

# ----------------------------------
# Experiment Configuration
# ----------------------------------
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)

ACTIONS = ["up", "down", "left", "right"]

ALPHA = 0.1
GAMMA = 0.9
EPISODES = 1000
MAX_STEPS = 100
EVALUATION_RUNS = 100

# Multiple independent random seeds
SEEDS = [11, 22, 33, 44, 55]

# Number of obstacle layouts per seed
LAYOUTS_PER_SEED = 5

# Keep the start and goal free
OBSTACLE_COUNT = 4


# ----------------------------------
# Environment
# ----------------------------------
def generate_obstacles(seed):
    rng = random.Random(seed)

    candidates = [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if (r, c) not in (START, GOAL)
    ]

    # Generate obstacles, but reject layouts where
    # the goal cannot be reached from the start.
    while True:
        obstacles = set(
            rng.sample(candidates, OBSTACLE_COUNT)
        )

        if is_reachable(obstacles):
            return obstacles


def is_reachable(obstacles):
    queue = [START]
    visited = {START}

    while queue:
        state = queue.pop(0)

        if state == GOAL:
            return True

        for action in ACTIONS:
            next_state = move(state, action, obstacles)

            if (
                next_state != state
                and next_state not in visited
            ):
                visited.add(next_state)
                queue.append(next_state)

    return False


def move(state, action, obstacles):
    row, col = state

    if action == "up":
        row -= 1
    elif action == "down":
        row += 1
    elif action == "left":
        col -= 1
    elif action == "right":
        col += 1

    next_state = (
        max(0, min(GRID_SIZE - 1, row)),
        max(0, min(GRID_SIZE - 1, col))
    )

    # An obstacle blocks movement
    if next_state in obstacles:
        return state

    return next_state


def valid_actions(state, obstacles):
    valid = []

    for action in ACTIONS:
        next_state = move(state, action, obstacles)

        if next_state != state:
            valid.append(action)

    # If surrounded, allow any action so the agent
    # can still receive a transition and penalty.
    return valid if valid else ACTIONS.copy()


# ----------------------------------
# Q-table and action selection
# ----------------------------------
def new_q_table():
    return {
        (r, c): {action: 0.0 for action in ACTIONS}
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
    }


def choose_action(q_table, state, obstacles, epsilon, rng):
    actions = valid_actions(state, obstacles)

    if rng.random() < epsilon:
        return rng.choice(actions)

    best_value = max(q_table[state][a] for a in actions)

    best_actions = [
        a for a in actions
        if q_table[state][a] == best_value
    ]

    return rng.choice(best_actions)


# ----------------------------------
# Training
# ----------------------------------
def train(obstacles, human_guided, seed):
    rng = random.Random(seed)
    q_table = new_q_table()

    successes = 0
    successful_steps = []
    episode_history = []

    for episode in range(EPISODES):
        state = START

        epsilon = max(
            0.05,
            1.0 - episode / (EPISODES * 0.8)
        )

        episode_success = 0
        episode_steps = MAX_STEPS
        total_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            action = choose_action(
                q_table, state, obstacles, epsilon, rng
            )

            next_state = move(state, action, obstacles)

            if next_state == GOAL:
                reward = 20.0
            elif next_state == state:
                reward = -3.0
            else:
                reward = -1.0

            # Simulated guidance, not actual human feedback
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

            actions_next = valid_actions(next_state, obstacles)

            if next_state == GOAL:
                future_q = 0.0
            else:
                future_q = max(
                    q_table[next_state][a]
                    for a in actions_next
                )

            old_q = q_table[state][action]

            q_table[state][action] = old_q + ALPHA * (
                reward + GAMMA * future_q - old_q
            )

            total_reward += reward
            state = next_state

            if state == GOAL:
                successes += 1
                episode_success = 1
                episode_steps = step
                successful_steps.append(step)
                break

        episode_history.append({
            "success": episode_success,
            "steps": episode_steps,
            "reward": total_reward
        })

    avg_steps = (
        statistics.mean(successful_steps)
        if successful_steps else None
    )

    return q_table, successes, avg_steps, episode_history


# ----------------------------------
# Evaluation
# ----------------------------------
def evaluate(q_table, obstacles, seed):
    rng = random.Random(seed)
    successes = 0
    successful_steps = []

    for _ in range(EVALUATION_RUNS):
        state = START

        for step in range(1, MAX_STEPS + 1):
            actions = valid_actions(state, obstacles)

            best_value = max(
                q_table[state][a] for a in actions
            )

            best_actions = [
                a for a in actions
                if q_table[state][a] == best_value
            ]

            action = rng.choice(best_actions)
            next_state = move(state, action, obstacles)

            # Safety guard against accidental cycles
            if next_state == state:
                break

            state = next_state

            if state == GOAL:
                successes += 1
                successful_steps.append(step)
                break

    success_rate = successes / EVALUATION_RUNS * 100

    avg_steps = (
        statistics.mean(successful_steps)
        if successful_steps else None
    )

    return success_rate, avg_steps


# ----------------------------------
# Run paired experiments
# ----------------------------------
def mean_or_none(values):
    return statistics.mean(values) if values else None


def std_or_zero(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def run_experiments():
    methods = {
        "Baseline Q-Learning": False,
        "Simulated Feedback-Guided Q-Learning": True
    }

    results = []
    learning_rows = []

    for seed in SEEDS:
        for layout_index in range(LAYOUTS_PER_SEED):
            layout_seed = seed * 100 + layout_index
            obstacles = generate_obstacles(layout_seed)

            print(
                f"\nSeed {seed}, layout {layout_index + 1}: "
                f"{sorted(obstacles)}"
            )

            for method_name, guided in methods.items():
                training_seed = seed * 10000 + layout_index * 100 + 7

                q_table, successes, avg_steps, history = train(
                    obstacles,
                    human_guided=guided,
                    seed=training_seed
                )

                eval_rate, eval_steps = evaluate(
                    q_table,
                    obstacles,
                    seed=training_seed + 50000
                )

                results.append({
                    "Seed": seed,
                    "Layout": layout_index + 1,
                    "Method": method_name,
                    "Obstacle Count": len(obstacles),
                    "Training Success Rate (%)":
                        successes / EPISODES * 100,
                    "Training Average Steps": avg_steps,
                    "Evaluation Success Rate (%)": eval_rate,
                    "Evaluation Average Steps": eval_steps
                })

                # Store learning-curve observations every 50 episodes
                for end in range(50, EPISODES + 1, 50):
                    window = history[end - 50:end]

                    learning_rows.append({
                        "Seed": seed,
                        "Layout": layout_index + 1,
                        "Method": method_name,
                        "Episode End": end,
                        "Success Rate (%)": (
                            statistics.mean(
                                row["success"] for row in window
                            ) * 100
                        ),
                        "Average Reward": statistics.mean(
                            row["reward"] for row in window
                        )
                    })

                print(
                    f"  {method_name}: "
                    f"training={successes / EPISODES * 100:.2f}%, "
                    f"evaluation={eval_rate:.2f}%, "
                    f"eval steps={eval_steps}"
                )

    # Save every individual layout result
    with open(
        "random_obstacle_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        fields = list(results[0].keys())
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    with open(
        "random_obstacle_learning_curves.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        fields = list(learning_rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(learning_rows)

    # Summary across paired layouts
    print("\n" + "=" * 65)
    print("SUMMARY ACROSS RANDOM OBSTACLE LAYOUTS")
    print("=" * 65)

    for method_name in methods:
        rows = [
            row for row in results
            if row["Method"] == method_name
        ]

        training = [r["Training Success Rate (%)"] for r in rows]
        evaluation = [r["Evaluation Success Rate (%)"] for r in rows]
        eval_steps = [
            r["Evaluation Average Steps"]
            for r in rows
            if r["Evaluation Average Steps"] is not None
        ]

        print(f"\n{method_name}")
        print(
            f"Training success: {mean_or_none(training):.2f}% "
            f"(SD {std_or_zero(training):.2f})"
        )
        print(
            f"Evaluation success: {mean_or_none(evaluation):.2f}% "
            f"(SD {std_or_zero(evaluation):.2f})"
        )
        print(
            "Evaluation steps among successful runs: "
            + (
                f"{mean_or_none(eval_steps):.2f}"
                if eval_steps else "N/A"
            )
        )

    print("\nSaved:")
    print("- random_obstacle_results.csv")
    print("- random_obstacle_learning_curves.csv")
    print("\nResearch limitation:")
    print("Guidance is simulated using distance to the goal.")
    print("It is not genuine human feedback.")


if __name__ == "__main__":
    run_experiments()
