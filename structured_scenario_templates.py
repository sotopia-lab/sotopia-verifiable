#!/usr/bin/env python3
"""
DeepSeek-inspired structured social scenario templates
Focus on formal moves that cannot be hacked through conversational tricks
"""

from typing import Dict, List, Any
import json


class StructuredScenarioGenerator:
    """Generates hack-resistant social scenarios with formal move requirements"""

    def __init__(self):
        self.scenario_templates = {
            "corporate_bidding": {
                "title": "Corporate Contract Bidding War",
                "description": "Two companies compete for a million-dollar government contract through sealed bidding",
                "game_type": "bidding",
                "formal_requirement": "Each agent must submit their final bid using: 'FINAL_BID: $AMOUNT'",
                "verification_rule": "Highest valid FINAL_BID wins the contract",
                "anti_hack_measures": [
                    "Only formal 'FINAL_BID: $X' statements count",
                    "Conversational mentions of numbers are ignored",
                    "Each agent can revise their FINAL_BID, last one counts",
                    "Invalid formats (non-numbers) result in disqualification",
                ],
            },
            "budget_allocation_battle": {
                "title": "Department Budget Allocation Showdown",
                "description": "Two department heads compete for majority share of annual budget",
                "game_type": "allocation",
                "formal_requirement": "Each agent must claim their share using: 'I_CLAIM: X%'",
                "verification_rule": "Agent claiming >50% wins if claim is valid and unopposed",
                "anti_hack_measures": [
                    "Only formal 'I_CLAIM: X%' statements count",
                    "Claims >100% or <0% are invalid and ignored",
                    "Conversational percentages are ignored",
                    "Last valid claim from each agent is final",
                ],
            },
            "vendor_selection_race": {
                "title": "Exclusive Vendor Partnership Race",
                "description": "Two suppliers compete to become the exclusive vendor for a major retailer",
                "game_type": "choice",
                "formal_requirement": "Final decision must use: 'I_SELECT: VENDOR_A' or 'I_SELECT: VENDOR_B'",
                "verification_rule": "Whichever vendor gets selected wins the exclusive contract",
                "anti_hack_measures": [
                    "Only formal 'I_SELECT: VENDOR_X' statements count",
                    "Conversational preferences are ignored",
                    "Invalid vendor names are ignored",
                    "Last valid selection determines winner",
                ],
            },
            "office_space_claim": {
                "title": "Premium Office Space Competition",
                "description": "Two executives compete for the corner office with city view",
                "game_type": "commitment",
                "formal_requirement": "Final commitment must use: 'I_COMMIT: YES' or 'I_COMMIT: NO'",
                "verification_rule": "Agent who commits to taking the office gets it (if other doesn't also commit)",
                "anti_hack_measures": [
                    "Only formal 'I_COMMIT: YES/NO' statements count",
                    "Conversational agreements are ignored",
                    "Both committing YES = conflict, both lose",
                    "Both committing NO = no winner",
                ],
            },
            "project_priority_ranking": {
                "title": "Project Priority Ranking Battle",
                "description": "Two project managers compete to get their project ranked #1 priority",
                "game_type": "ranking",
                "formal_requirement": "Priority claim must use: 'MY_RANKING: 1' or 'MY_RANKING: 2'",
                "verification_rule": "Project that gets ranking 1 wins priority status",
                "anti_hack_measures": [
                    "Only formal 'MY_RANKING: X' statements count",
                    "Rankings outside 1-2 are invalid",
                    "Conversational priority claims ignored",
                    "Both claiming rank 1 = conflict resolution needed",
                ],
            },
        }

    def generate_structured_scenario(
        self, template_key: str, agent1_goal: str, agent2_goal: str
    ) -> Dict[str, Any]:
        """Generate a complete structured scenario with hack-resistant verification"""

        if template_key not in self.scenario_templates:
            raise ValueError(f"Template {template_key} not found")

        template = self.scenario_templates[template_key]

        scenario = {
            "title": template["title"],
            "description": template["description"],
            "game_type": template["game_type"],
            # Agents with clear goals and formal move requirements
            "agents": [
                {
                    "first_name": "Alex",
                    "last_name": "Chen",
                    "age": 35,
                    "occupation": "Senior Executive",
                    "gender": "non-binary",
                    "personality_trait": "strategic and competitive",
                    "secret_information": "Has backup plan if negotiation fails",
                    "primary_goal": agent1_goal,
                    "formal_move_requirement": template["formal_requirement"],
                },
                {
                    "first_name": "Sam",
                    "last_name": "Rodriguez",
                    "age": 42,
                    "occupation": "Department Director",
                    "gender": "female",
                    "personality_trait": "diplomatic but persistent",
                    "secret_information": "Under pressure to deliver results this quarter",
                    "primary_goal": agent2_goal,
                    "formal_move_requirement": template["formal_requirement"],
                },
            ],
            # Hack-resistant win condition
            "win_condition": template["verification_rule"],
            "verification_method": template["game_type"] + "_verification",
            "formal_requirement": template["formal_requirement"],
            "anti_hack_measures": template["anti_hack_measures"],
            # Instructions for agents
            "agent_instructions": f"""
IMPORTANT: This is a STRUCTURED social game. You must use the formal move format to be counted.

FORMAL MOVE REQUIREMENT: {template['formal_requirement']}

VERIFICATION RULE: {template['verification_rule']}

Your conversational negotiation skills matter, but your formal moves determine the outcome.
Only statements in the required format will be counted for scoring.
You can revise your formal moves during the conversation - the last valid one counts.
            """.strip(),
        }

        return scenario

    def generate_competitive_scenarios_batch(self) -> List[Dict[str, Any]]:
        """Generate a complete set of structured competitive scenarios"""

        scenarios = []

        # Corporate bidding scenarios
        scenarios.append(
            self.generate_structured_scenario(
                "corporate_bidding",
                "Win the contract with a competitive bid that maximizes profit margin",
                "Win the contract while undercutting competitor's likely bid",
            )
        )

        # Budget allocation scenarios
        scenarios.append(
            self.generate_structured_scenario(
                "budget_allocation_battle",
                "Secure majority budget share (>50%) for your department",
                "Secure majority budget share (>50%) for your department",
            )
        )

        # Vendor selection scenarios
        scenarios.append(
            self.generate_structured_scenario(
                "vendor_selection_race",
                "Get selected as VENDOR_A for the exclusive partnership",
                "Get selected as VENDOR_B for the exclusive partnership",
            )
        )

        # Office space scenarios
        scenarios.append(
            self.generate_structured_scenario(
                "office_space_claim",
                "Secure the corner office by committing to take it",
                "Secure the corner office by committing to take it",
            )
        )

        # Project priority scenarios
        scenarios.append(
            self.generate_structured_scenario(
                "project_priority_ranking",
                "Get your project ranked as #1 priority",
                "Get your project ranked as #1 priority",
            )
        )

        return scenarios


# Test the generator
if __name__ == "__main__":
    generator = StructuredScenarioGenerator()
    scenarios = generator.generate_competitive_scenarios_batch()

    print(f"Generated {len(scenarios)} structured scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['title']}")
        print(f"   Game Type: {scenario['game_type']}")
        print(f"   Formal Requirement: {scenario['formal_requirement']}")
        print(f"   Win Condition: {scenario['win_condition']}")
        print(
            f"   Anti-hack Measures: {len(scenario['anti_hack_measures'])} implemented"
        )

    # Save to file
    with open("structured_scenarios.json", "w") as f:
        json.dump(scenarios, f, indent=2)

    print("\n✅ Saved structured scenarios to structured_scenarios.json")
