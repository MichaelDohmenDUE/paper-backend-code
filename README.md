## Repository Structure

The codebase is ordered by algorithm types:

* `ActionValue/`: Implementations of DQN, DDQN and Dueling DQN
* `DeterministicPolicy/`: Deterministic policy gradient methods (DDPG, TD3).
* `StochasticPolicy/`: REINFORCE, REINFORCE_BASELINE, custom Advantage Actor Critic, PPO
* `Utils/`: buffers, environment wrappers, and Node Library
* `TobeTranslatedALgorithms/`: Algorithms that could not be converted to a signal flow graph yet, not relevant for the Thesis but it is important for Fatih later so it won't get deleterd

The structure is inspired by CleanRL, which means that each algorithm has its own main script and a `src/` Folder that contains the  
contains den Agent Networks, the Training Processor and the Data Coellction Proccessor at the very least.
The main script is named after the algorithm.
## Requirements
This project uses [`uv`] for fast, deterministic package management.

To set up the environment and install dependencies, follow this guide for Linux/Mac Machines:
```bash
# 1. Install uv (if not already installed)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
# 2. Open the Backend folder
cd <yourfolder>/backend
# 3. Create a virtual environment and install dependencies from the lockfile
uv sync
```
This project uses wandb (https://wandb.ai/site/) in order to track and follow 
the logging on the Training Scripts. You can make a free account on their website.
It is currently mandatory for logging.

Copy the ".env.example" to a ".env" file and insert your WandB username.

Then you can then start the Training Scripts with
```
PYTHONPATH=.. uv run ActionValue/DQN/DQN.py
```
This is necessary, as the backend is part of the bigger DEEPRLWebPLayGround. The scripts support GPU usage.
Using your GPU will speed up Training somewhat but the scripts are very CPU heavy. I would also advide to be careful with the Atari scripts for the Action-Value Variants.
They currently use a lot of memory, so you might beed to adapt the Hyperparamters on a weaker machine. 32 GB of Ram is recommended to run the scripts comfortably.

Wandb can be run in an offline or online modus. By default the online modus will be tunred on and local runs will be 
saved to a wandb folder within the algorithm folder. You can turn on offline modus with
```
wandb offline
```

Each Training Script has its own set Hyperparameters that can be adapted in order to customize your Training.


