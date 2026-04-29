import wandb
import numpy as np

api = wandb.Api()
entity = "michael_dohmen-"
project = "my-ppo-benchmarks"

runs = api.runs(f"{entity}/{project}", filters={"display_name": {"$regex": "PPO_discrete_atari-seed-*"}})

final_rewards = []

for run in runs:
    if "eval/avg_reward" in run.summary:
        final_rewards.append(run.summary["eval/avg_reward"])
        print(final_rewards)
    else:
        print(f"Metric not found for run: {run.name}")

if final_rewards:
    mean_val = np.mean(final_rewards)
    std_val = np.std(final_rewards)

    print(f"Statistics for {len(final_rewards)} seeds:")
    print(f"Mean:    {mean_val:.2f}")
    print(f"Std Dev: {std_val:.2f}")
else:
    print("No matching metrics found. Ensure the filter and metric names are correct.")