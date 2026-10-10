
import csv
import matplotlib.pyplot as plt

methods = []
training_success_rates = []
training_avg_steps = []
evaluation_success_rates = []
evaluation_avg_steps = []

with open("experiment_results.csv", "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        methods.append(row["Method"])
        training_success_rates.append(
            float(row["Training Success Rate (%)"])
        )
        training_avg_steps.append(
            float(row["Training Average Steps"])
        )
        evaluation_success_rates.append(
            float(row["Evaluation Success Rate (%)"])
        )
        evaluation_avg_steps.append(
            float(row["Evaluation Average Steps"])
        )

# Graph 1: Training success rate
plt.figure(figsize=(9, 5))
plt.bar(methods, training_success_rates)
plt.ylabel("Training Success Rate (%)")
plt.title("Training Success Rate Comparison")
plt.ylim(0, 100)
plt.xticks(rotation=10, ha="right")
plt.tight_layout()
plt.savefig("training_success_comparison.png", dpi=300)
plt.show()

# Graph 2: Average training steps
plt.figure(figsize=(9, 5))
plt.bar(methods, training_avg_steps)
plt.ylabel("Average Steps")
plt.title("Average Training Steps Comparison")
plt.xticks(rotation=10, ha="right")
plt.tight_layout()
plt.savefig("training_steps_comparison.png", dpi=300)
plt.show()

# Graph 3: Evaluation success rate
plt.figure(figsize=(9, 5))
plt.bar(methods, evaluation_success_rates)
plt.ylabel("Evaluation Success Rate (%)")
plt.title("Evaluation Success Rate Comparison")
plt.ylim(0, 100)
plt.xticks(rotation=10, ha="right")
plt.tight_layout()
plt.savefig("evaluation_success_comparison.png", dpi=300)
plt.show()

print("All graphs saved successfully!")
