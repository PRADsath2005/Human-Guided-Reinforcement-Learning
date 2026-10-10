
import random

GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)

ACTIONS = ["up", "down", "left", "right"]

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.2
EPISODES = 1000
MAX_STEPS = 100

# Q-table: each position has a value for every action
q_table = {
    (r, c): {action: 0.0 for action in ACTIONS}
    for r in range(GRID_SIZE)
    for c in range(GRID_SIZE)
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

def choose_action(position):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)

    values = q_table[position]
    best_value = max(values.values())
    best_actions = [
        action for action, value in values.items()
        if value == best_value
    ]
    return random.choice(best_actions)

# Train the agent
for episode in range(EPISODES):
    position = START

    for step in range(MAX_STEPS):
        action = choose_action(position)
        next_position = move(position, action)

        reward = 100 if next_position == GOAL else -1

        old_value = q_table[position][action]
        future_value = max(q_table[next_position].values())

        q_table[position][action] = old_value + LEARNING_RATE * (
            reward + DISCOUNT_FACTOR * future_value - old_value
        )

        position = next_position

        if position == GOAL:
            break

# Test the trained agent without random exploration
position = START
steps = 0

print("Q-Learning Training Completed!")
print("Start:", START)
print("Goal:", GOAL)
print("\nTesting the trained agent:")

while position != GOAL and steps < MAX_STEPS:
    values = q_table[position]
    best_value = max(values.values())
    best_actions = [
        action for action, value in values.items()
        if value == best_value
    ]
    action = random.choice(best_actions)

    position = move(position, action)
    steps += 1

    print(f"Step {steps}: {action} -> {position}")

if position == GOAL:
    print(f"\nSUCCESS! Goal reached in {steps} steps.")
else:
    print("\nGoal not reached. Train again.")
