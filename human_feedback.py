
import random

GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)

ACTIONS = ["up", "down", "left", "right"]

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
MAX_STEPS = 50

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

def create_q_table():
    return {
        (r, c): {action: 0.0 for action in ACTIONS}
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
    }

def choose_action(q_table, position, epsilon=0.2):
    actions = valid_actions(position)

    if random.random() < epsilon:
        return random.choice(actions)

    best_value = max(q_table[position][a] for a in actions)
    best_actions = [
        a for a in actions
        if q_table[position][a] == best_value
    ]
    return random.choice(best_actions)

def get_feedback():
    while True:
        choice = input(
            "Feedback: 1=Good, 2=Bad, 3=Skip: "
        ).strip()

        if choice == "1":
            return 5.0
        if choice == "2":
            return -5.0
        if choice == "3":
            return 0.0

        print("Please enter 1, 2, or 3.")

def train():
    q_table = create_q_table()
    position = START

    print("Human-Guided Q-Learning")
    print("Start:", START, "| Goal:", GOAL)

    for step in range(MAX_STEPS):
        print("\nCurrent position:", position)

        action = choose_action(q_table, position)
        next_position = move(position, action)

        print("Suggested action:", action)
        print("Next position:", next_position)

        human_reward = get_feedback()
        environment_reward = 20.0 if next_position == GOAL else -1.0
        total_reward = human_reward + environment_reward

        old_value = q_table[position][action]
        future_value = (
            0.0 if next_position == GOAL
            else max(q_table[next_position][a]
                     for a in valid_actions(next_position))
        )

        q_table[position][action] = old_value + LEARNING_RATE * (
            total_reward + DISCOUNT_FACTOR * future_value - old_value
        )

        print(f"Human reward: {human_reward}")
        print(f"Environment reward: {environment_reward}")
        print(f"Updated Q-value: {q_table[position][action]:.2f}")

        position = next_position

        if position == GOAL:
            print(f"\nGoal reached in {step + 1} steps!")
            return q_table

    print("\nStep limit reached. Try another run.")
    return q_table

if __name__ == "__main__":
    learned_q_table = train()
    print("\nTraining session completed.")
