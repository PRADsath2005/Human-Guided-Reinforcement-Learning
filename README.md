# Human-Guided Reinforcement Learning Using Q-Learning

## Project Overview

This project demonstrates a Human-Guided Reinforcement Learning approach using the Q-Learning algorithm. It compares standard Q-Learning with a feedback-guided approach in a grid-world environment.

The agent learns to navigate from a starting position to a goal by selecting actions and receiving rewards.

## Objectives

- Implement a baseline Q-Learning algorithm.
- Simulate human feedback to guide the agent's learning.
- Compare training performance between both approaches.
- Evaluate success rates and average steps.
- Visualize experimental results using graphs.

## Technologies Used

- Python
- Matplotlib
- CSV
- Q-Learning
- Reinforcement Learning

## Project Structure

```text
Human_Guided_RL/
├── main.py
├── q_learning_backup.py
├── human_feedback.py
├── experiment.py
├── plot_results.py
├── experiment_results.csv
├── training_success_comparison.png
├── training_steps_comparison.png
├── evaluation_success_comparison.png
└── README.md
```

## Methodology

### 1. Baseline Q-Learning

The agent learns a policy through rewards received from the environment without additional human guidance.

### 2. Feedback-Guided Q-Learning

The agent receives simulated feedback that influences its learning process. The experiment uses distance-based simulated guidance, not feedback collected from real human participants.

### 3. Experimental Evaluation

Both approaches are evaluated using training success rate, average training steps, evaluation success rate, and average evaluation steps.

## Experimental Results

The experiment results are stored in `experiment_results.csv`.

The following graphs visualize the comparison:

### Training Success Rate

![Training Success Rate](training_success_comparison.png)

### Average Training Steps

![Average Training Steps](training_steps_comparison.png)

### Evaluation Success Rate

![Evaluation Success Rate](evaluation_success_comparison.png)

## How to Run

1. Install Python 3.
2. Install Matplotlib:

   ```bash
   python -m pip install matplotlib
   ```

3. Run the experiment:

   ```bash
   python experiment.py
   ```

4. Generate comparison graphs:

   ```bash
   python plot_results.py
   ```

## Conclusion

This project compares baseline Q-Learning with a simulated feedback-guided learning approach in a grid-world environment. The results help analyze how simulated guidance affects learning performance.

**Note:** Results are specific to the implemented environment and experimental configuration. They should not be interpreted as proof that real human feedback always improves reinforcement learning.
