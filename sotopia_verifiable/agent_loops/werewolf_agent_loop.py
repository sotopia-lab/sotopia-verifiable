# Copyright 2025 Sotopia Verifiable
# Werewolf Game Agent Loop for Multi-turn RL Training

import asyncio
import copy
import json
import logging
import os
import sys
from typing import Any, Optional
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.utils.profiler import simple_timer

# Add Sotopia werewolf to path
current_dir = os.path.dirname(os.path.abspath(__file__))
werewolf_dir = os.path.abspath(
    os.path.join(current_dir, "../../dependencies/sotopia/examples/experimental/games/werewolves")
)
if werewolf_dir not in sys.path:
    sys.path.insert(0, werewolf_dir)

from main import prepare_scenario, load_config, CONFIG_PATH
from sotopia.messages import AgentAction
from sotopia.generation_utils.generate import fill_template

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("werewolf_agent")
class WerewolfAgentLoop(AgentLoopBase):
    """
    Agent loop for Werewolf game training.

    Receives scenario metadata from dataset, initializes the game using
    Sotopia's game engine, and runs the trainee through the game.
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor, **kwargs)
        config = trainer_config.config

        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.max_game_turns = config.actor_rollout_ref.rollout.get("max_game_turns", 40)

        # Load base game config
        self._base_config = load_config(CONFIG_PATH)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run the werewolf game and collect trajectory for training."""
        metrics = {"generate_sequences": 0.0, "tool_calls": 0.0}
        request_id = uuid4().hex

        # Extract scenario from dataset fields (passed directly as kwargs)
        agents_config = kwargs.get("agents", [])
        trainee_name = kwargs.get("trainee_name")

        if not agents_config or not trainee_name:
            logger.error("Missing agents or trainee_name in scenario config")
            return self._empty_output(metrics)

        # Initialize game
        game_config = copy.deepcopy(self._base_config)
        game_config["agents"] = agents_config

        agent_models = {a["name"]: a.get("agent_model", "gpt-4o") for a in agents_config}
        role_map = {a["name"]: a["role"] for a in agents_config}

        # Create environment and agents
        env, agents_list = prepare_scenario("gpt-4o", agent_models, game_config)
        agents = {agent.agent_name: agent for agent in agents_list}
        trainee_agent = agents.get(trainee_name)

        # Reset environment
        obs_dict = env.reset(agents=agents)

        # Trajectory collection
        all_prompt_ids = []
        all_response_ids = []
        all_response_mask = []
        num_turns = 0

        # Game loop
        game_done = False
        turn = 0

        while not game_done and turn < self.max_game_turns:
            turn += 1

            # Check if trainee is active
            agent_names = list(env.agents)
            trainee_idx = agent_names.index(trainee_name) if trainee_name in agent_names else -1
            trainee_active = (
                trainee_idx >= 0
                and trainee_idx < len(env.action_mask)
                and env.action_mask[trainee_idx] == 1
            )

            # Collect actions from all agents
            actions = {}

            for agent_name in env.agents:
                agent_obs = obs_dict.get(agent_name)

                if agent_name == trainee_name and trainee_active:
                    # Trainee's turn - use verl's LLM server
                    with simple_timer("generate_sequences", metrics):
                        prompt, prompt_ids = await self._build_prompt(
                            trainee_agent, trainee_name, agent_obs
                        )

                        output = await self.server_manager.generate(
                            request_id=request_id,
                            prompt_ids=prompt_ids,
                            sampling_params=sampling_params,
                        )

                        response_ids = output.token_ids
                        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                        action = self._parse_action(response_text)

                        # Add to trajectory (only trainee's tokens)
                        if not all_prompt_ids:  # First turn
                            all_prompt_ids = prompt_ids
                        all_response_ids.extend(response_ids)
                        all_response_mask.extend([1] * len(response_ids))
                        num_turns += 1

                        actions[agent_name] = action
                else:
                    # Opponent - use deterministic action for now
                    # TODO: optionally use real LLM calls via agent.aact()
                    role = role_map.get(agent_name, "Villager")
                    actions[agent_name] = self._get_deterministic_action(agent_name, role, env)

            # Step environment
            obs_dict, rewards, terminated, truncated, info = await env.astep(actions)
            game_done = all(terminated.values())

            # Add environment feedback to trajectory (masked out)
            if trainee_active and not game_done:
                trainee_obs = obs_dict.get(trainee_name)
                if trainee_obs:
                    feedback_text = trainee_obs.to_natural_language()
                    if trainee_obs.action_instruction:
                        feedback_text += f"\n{trainee_obs.action_instruction}"
                    feedback_ids = self.tokenizer.encode(feedback_text, add_special_tokens=False)
                    all_response_ids.extend(feedback_ids)
                    all_response_mask.extend([0] * len(feedback_ids))  # Mask env feedback

        # Get final reward for trainee
        trainee_info = info.get(trainee_name, {})
        reward = trainee_info.get("complete_rating", 0.0)

        # Ensure we have some trajectory
        if not all_prompt_ids:
            return self._empty_output(metrics)

        return AgentLoopOutput(
            prompt_ids=all_prompt_ids,
            response_ids=all_response_ids[:self.response_length],
            response_mask=all_response_mask[:self.response_length],
            reward_score=reward,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields={
                "trainee_name": trainee_name,
                "trainee_role": role_map.get(trainee_name),
                "game_turns": turn,
            },
        )

    async def _build_prompt(
        self,
        trainee_agent,
        trainee_name: str,
        observation,
    ) -> tuple[str, list[int]]:
        """Build prompt using game engine's template system."""
        if trainee_agent and hasattr(trainee_agent, 'custom_template') and observation:
            # Use Sotopia's fill_template (same as LLMAgent.aact does)
            template = fill_template(
                trainee_agent.custom_template,
                action_instructions=observation.action_instruction or ""
            )
            prompt = fill_template(
                template,
                agent=trainee_name,
                history=observation.to_natural_language(),
                action_list=", ".join(observation.available_actions),
                turn_number=str(observation.turn_number),
                format_instructions='{"action_type": "<action_type>", "argument": "<argument>"}'
            )
        else:
            # Fallback
            prompt = observation.to_natural_language() if observation else "[No observation]"

        # Tokenize
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.encode(prompt, add_special_tokens=True)
        )

        return prompt, prompt_ids

    def _parse_action(self, action_str: str) -> AgentAction:
        """Parse LLM output into AgentAction."""
        try:
            clean_str = action_str.strip()
            if clean_str.startswith("```json"):
                clean_str = clean_str[7:]
            elif clean_str.startswith("```"):
                clean_str = clean_str[3:]
            if clean_str.endswith("```"):
                clean_str = clean_str[:-3]
            clean_str = clean_str.strip()

            data = json.loads(clean_str)
            if isinstance(data, dict):
                action_type = data.get("action_type", "none")
                argument = data.get("argument", "")
                if action_type in {"none", "speak", "action", "leave", "non-verbal communication"}:
                    return AgentAction(action_type=action_type, argument=argument)
        except Exception:
            pass

        # Invalid format - return "none"
        logger.warning(f"Invalid action format: {action_str[:100]}")
        return AgentAction(action_type="none", argument="")

    def _get_deterministic_action(self, agent_name: str, role: str, env) -> AgentAction:
        """Get deterministic action for opponent agents."""
        state_name = env.current_state if hasattr(env, 'current_state') else ""

        alive_others = [n for n, alive in env.agent_alive.items() if alive and n != agent_name]
        target = alive_others[0] if alive_others else agent_name

        if "Night_werewolf" in state_name and role == "Werewolf":
            return AgentAction(action_type="action", argument=f"kill {target}")
        elif "Night_seer" in state_name and role == "Seer":
            return AgentAction(action_type="action", argument=f"inspect {target}")
        elif "Night_witch" in state_name and role == "Witch":
            return AgentAction(action_type="none", argument="")
        elif "Day_discussion" in state_name:
            return AgentAction(action_type="speak", argument="I'm observing.")
        elif "Day_vote" in state_name:
            return AgentAction(action_type="action", argument=f"vote {target}")
        else:
            return AgentAction(action_type="none", argument="")

    def _empty_output(self, metrics: dict) -> AgentLoopOutput:
        """Return empty output when game fails to start."""
        return AgentLoopOutput(
            prompt_ids=[self.tokenizer.bos_token_id or 0],
            response_ids=[self.tokenizer.eos_token_id or 0],
            response_mask=[1],
            reward_score=0.0,
            num_turns=0,
            metrics=AgentLoopMetrics(**metrics),
        )
