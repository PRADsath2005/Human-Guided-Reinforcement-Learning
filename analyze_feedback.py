
import csv
from collections import Counter
import matplotlib.pyplot as plt

with open(
    "human_feedback_log.csv",
    "r",
    newline="",
    encoding="utf-8"
) as file:
    rows = list(csv.DictReader(file))

if not rows:
    print("No feedback records found.")
    raise SystemExit

feedback_counts = Counter(
    row["Feedback"] for row in rows
)

action_counts = Counter(
    row["Action"] for row in rows
)

total_reward = sum(
    float(row["Reward"]) for row in rows
)

print("\n=== HUMAN FEEDBACK ANALYSIS ===")
print("Total feedback:", len(rows))
print("Good feedback:", feedback_counts["good"])
print("Bad feedback:", feedback_counts["bad"])
print(f"Total reward: {total_reward:.2f}")

# Graph 1: Good vs Bad feedback
labels = ["Good", "Bad"]
counts = [
    feedback_counts["good"],
    feedback_counts["bad"]
]

plt.figure(figsize=(7, 5))
plt.bar(labels, counts)
plt.ylabel("Number of Feedback Records")
plt.title("Human Feedback Distribution")
plt.tight_layout()
plt.savefig("human_feedback_distribution.png", dpi=300)
plt.close()

# Graph 2: Feedback by action
actions = ["up", "down", "left", "right"]
action_values = [action_counts[action] for action in actions]

plt.figure(figsize=(7, 5))
plt.bar(actions, action_values)
plt.ylabel("Number of Feedback Records")
plt.title("Human Feedback Actions")
plt.tight_layout()
plt.savefig("human_feedback_actions.png", dpi=300)
plt.close()

# Graph 3: Reward distribution
plt.figure(figsize=(6, 6))
plt.pie(
    counts,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Good vs Bad Feedback (%)")
plt.tight_layout()
plt.savefig("human_feedback_pie.png", dpi=300)
plt.close()

print("\nSaved graphs:")
print("human_feedback_distribution.png")
print("human_feedback_actions.png")
print("human_feedback_pie.png")
