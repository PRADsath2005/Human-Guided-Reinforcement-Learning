
import csv
import statistics
import matplotlib.pyplot as plt

with open(
    "random_obstacle_results.csv",
    "r",
    newline="",
    encoding="utf-8"
) as file:
    rows = list(csv.DictReader(file))

methods = sorted(set(row["Method"] for row in rows))

metrics = [
    ("Training Success Rate (%)",
     "Training Success Rate (%)",
     "random_obstacle_training_error_bars.png"),

    ("Evaluation Success Rate (%)",
     "Evaluation Success Rate (%)",
     "random_obstacle_evaluation_error_bars.png")
]

for column, title, filename in metrics:
    means = []
    errors = []

    for method in methods:
        values = [
            float(row[column])
            for row in rows
            if row["Method"] == method
        ]

        means.append(statistics.mean(values) if values else 0)

        errors.append(
            statistics.stdev(values)
            if len(values) > 1 else 0
        )

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        methods,
        means,
        yerr=errors,
        capsize=6
    )

    plt.ylabel(title)
    plt.title(title + " Across Random Obstacle Layouts")
    plt.ylim(0, 105)
    plt.xticks(rotation=10, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, value in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.2f}%",
            ha="center"
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print("Saved:", filename)
    plt.show()

print("Both comparison graphs created.")
