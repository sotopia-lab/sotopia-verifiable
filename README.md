# Sotopia-Verifiable: Werewolf Training Pipeline

This repository integrates the **Sotopia** Werewolf environment with the **Verl** Reinforcement Learning framework to train a Qwen agent using PPO.

## 🏗 Architecture Overview

We use **Verl's Async Agent Loop** architecture. This differs from standard RLHF (which typically does single-turn Q&A) by moving the entire game interaction inside the inference worker.

### The Pipeline Flow
1.  **Orchestrator (`train_werewolf.py`)**:
    *   Starts the Ray cluster.
    *   Loads the PPO config (`werewolf_ppo.yaml`).
    *   Distributes work to Ray workers.

2.  **Make Work (`data/werewolf_train.parquet`)**:
    *   Verl is data-driven. It asks "What prompts should I train on?".
    *   Since Werewolf is an environment we just "start", we feed Verl a **Dummy Dataset** (created by `create_dummy_data.py`).
    *   Each row in the .parquet file triggers one game episode.

3.  **Rollout Worker (vLLM + AgentLoop)**:
    *   Verl sends a "prompt" (e.g., "Game 1") to the vLLM worker.
    *   Instead of just generating text, the worker triggers our **Custom Agent Loop** (`werewolf_agent_loop.py`).

4.  **The Game Loop (`WerewolfAgentLoop`)**:
    *   **Initializes**: Calls `SotopiaWerewolfWrapper` to create a fresh Werewolf game.
    *   **Interacts**:
        *   **Action**: Uses vLLM `generate()` to get the Trainee's move.
        *   **Environment**: Calls `wrapper.step()` to process that move and simulate other agents (using GPT-4 or other policies defined in Sotopia).
        *   **Loop**: Repeats until the game ends.
    *   **Returns**: A full trajectory of (Prompt, Response, Reward) tokens back to the PPO trainer.

## 📂 Key Files Explained

### 1. `examples/train_werewolf.py`
The entry point. It registers our custom `werewolf` worker type so Verl knows it exists, then launches the standard Verl PPO main function.

### 2. `config/werewolf_ppo.yaml`
The main configuration file (Hydra format). Key settings:
*   `rollout.name: vllm`: We use vLLM for fast inference.
*   `rollout.mode: async`: CRITICAL. Tells Verl we are using the Agent Loop system.
*   `rollout.agent.default_agent_loop: werewolf`: Points to our custom class.

### 3. `config/agent_loop/werewolf.yaml`
A small config file that maps the name "werewolf" to the Python class `DidacticAgentLoop`. Loaded dynamically by the AgentLoopManager.

### 4. `sotopia_verifiable/workers/werewolf_agent_loop.py`
**The Core Logic**. This class inherits from `AgentLoopBase`.
*   It manages the conversation history (User/Assistant turns).
*   It handles the masking (we only train on the Agent's output, not the Environment's observations).
*   It calculates the final reward.

### 5. `sotopia_verifiable/envs/werewolf_env.py`
**The Bridge**. Sotopia is a complex multi-agent system. This wrapper makes it look like a simple environment:
*   `setup_game()`: Creates a scenario.
*   `step()`: Handles the Trainee's action and automatically runs all other agents (Imposters, Villagers) to finish the round.
*   `_parse_action()`: Ensures the LLM's text output ("I vote for X") becomes a valid game action.

### 6. `examples/create_dummy_data.py`
Generates `data/werewolf_train.parquet`.
*   **Why?** Verl requires an input dataset to define the "epoch".
*   We generate 100 "dummy" items. This effectively means "Run 100 parallel game episodes per epoch".

## 🚀 How to Run

1.  **Generate Data**:
    ```bash
    python examples/create_dummy_data.py
    ```
2.  **Start Training**:
    ```bash
    python examples/train_werewolf.py --config-path ../config --config-name werewolf_ppo
    ```
