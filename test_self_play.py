#!/usr/bin/env python3
"""
Test script for self-play evaluation framework

Quick test to verify the self-play system works before running full training.
"""

import asyncio
import logging

from self_play_evaluator import SelfPlayEvaluator


async def main():
    logging.basicConfig(level=logging.INFO)

    print("🧪 Testing Self-Play Evaluation Framework")
    print("=" * 50)

    # Test with base model (no checkpoint)
    evaluator = SelfPlayEvaluator(
        trainee_model_path=None,  # Use base model
        partner_model="gpt-4o",
    )

    print("Loading scenarios...")
    scenarios = evaluator.load_all_scenarios()
    print(f"✅ Loaded {len(scenarios)} scenarios")

    print("\nTesting single game simulation...")
    test_scenario = scenarios[0]  # Use first scenario
    result = await evaluator.simulate_game(test_scenario)

    print("✅ Game completed:")
    print(f"   Scenario: {result.scenario_title}")
    print(f"   Trainee won: {result.trainee_won}")
    print(f"   Partner won: {result.partner_won}")
    print(f"   Turns: {result.conversation_turns}")
    print(f"   Outcome: {result.game_outcome}")

    if result.conversation_log:
        print("\n💬 Sample conversation:")
        # for turn in result.conversation_log[:3]:  # Show first 3 turns
        #     agent_type = "🤖" if turn.get('agent_type') == 'trainee' else "👤"
        #     print(f"   {agent_type} {turn['agent']}: {turn['argument'][:80]}...")
        for turn in result.conversation_log[:]:  # Show all turns
            agent_type = "🤖" if turn.get("agent_type") == "trainee" else "👤"
            print(f"   {agent_type} {turn['agent']}: {turn['argument'][:]}")

    print(
        f"\n🎯 Win condition verification working: {'✅' if not result.error_msg else '❌'}"
    )

    print("\n🎉 Self-play framework test completed!")
    print("Ready for full training pipeline.")


if __name__ == "__main__":
    asyncio.run(main())
