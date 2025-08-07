#!/usr/bin/env python3
"""
DeepSeek-inspired structured social verification system
Focuses on formal "moves" rather than conversation parsing to prevent reward hacking
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GameType(Enum):
    BIDDING = "bidding"
    ALLOCATION = "allocation"
    CHOICE = "choice"
    RANKING = "ranking"
    COMMITMENT = "commitment"


@dataclass
class FormalMove:
    """Represents a structured, verifiable move in a social game"""

    agent: str
    move_type: str
    value: Any
    turn_number: int
    raw_text: str


@dataclass
class GameResult:
    """Results of structured social game verification"""

    winner: Optional[str]  # 'agent1', 'agent2', or None (draw/invalid)
    loser: Optional[str]
    reason: str
    valid_game: bool
    agent1_moves: List[FormalMove]
    agent2_moves: List[FormalMove]


class StructuredSocialVerifier:
    """
    DeepSeek-inspired verifier that looks for structured moves, not conversational tricks
    """

    def __init__(self):
        # Define required move formats (like DeepSeek's <think></think> tags)
        self.move_patterns = {
            GameType.BIDDING: {
                "pattern": r"FINAL_BID:\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)",
                "description": "Agent must say: 'FINAL_BID: $50000'",
            },
            GameType.ALLOCATION: {
                "pattern": r"I_CLAIM:\s*(\d+(?:\.\d+)?)%",
                "description": "Agent must say: 'I_CLAIM: 65%'",
            },
            GameType.CHOICE: {
                "pattern": r"I_SELECT:\s*([A-Z]+)",
                "description": "Agent must say: 'I_SELECT: OPTION_A'",
            },
            GameType.RANKING: {
                "pattern": r"MY_RANKING:\s*(\d+)",
                "description": "Agent must say: 'MY_RANKING: 1'",
            },
            GameType.COMMITMENT: {
                "pattern": r"I_COMMIT:\s*([YES|NO]+)",
                "description": "Agent must say: 'I_COMMIT: YES'",
            },
        }

    def extract_formal_moves(
        self, conversation: List[Dict[str, Any]], game_type: GameType
    ) -> Tuple[List[FormalMove], List[FormalMove]]:
        """Extract structured moves from conversation (immune to conversational tricks)"""

        pattern_info = self.move_patterns[game_type]
        pattern = pattern_info["pattern"]

        agent1_moves = []
        agent2_moves = []

        for turn_idx, turn in enumerate(conversation):
            if turn.get("action_type") != "speak":
                continue

            agent = turn.get("agent", "unknown")
            text = turn.get("argument", "")

            # Extract formal moves using regex (like DeepSeek's format verification)
            matches = re.findall(pattern, text, re.IGNORECASE)

            for match in matches:
                move = FormalMove(
                    agent=agent,
                    move_type=game_type.value,
                    value=match,
                    turn_number=turn_idx,
                    raw_text=text,
                )

                if agent == "agent1":
                    agent1_moves.append(move)
                elif agent == "agent2":
                    agent2_moves.append(move)

        return agent1_moves, agent2_moves

    def verify_bidding_game(self, conversation: List[Dict[str, Any]]) -> GameResult:
        """
        Bidding verification: Highest valid bid wins
        Anti-hack: Only counts formal 'FINAL_BID: $X' statements
        """
        agent1_moves, agent2_moves = self.extract_formal_moves(
            conversation, GameType.BIDDING
        )

        # Get final (last) valid bid from each agent
        agent1_final_bid = None
        agent2_final_bid = None

        if agent1_moves:
            # Parse final bid amount
            final_move = agent1_moves[-1]
            try:
                agent1_final_bid = float(final_move.value.replace(",", ""))
            except (ValueError, AttributeError):
                logger.warning(f"Invalid bid format from agent1: {final_move.value}")

        if agent2_moves:
            final_move = agent2_moves[-1]
            try:
                agent2_final_bid = float(final_move.value.replace(",", ""))
            except (ValueError, AttributeError):
                logger.warning(f"Invalid bid format from agent2: {final_move.value}")

        # Determine winner based on formal bids only
        if agent1_final_bid is None and agent2_final_bid is None:
            return GameResult(
                winner=None,
                loser=None,
                reason="No valid FINAL_BID statements found",
                valid_game=False,
                agent1_moves=agent1_moves,
                agent2_moves=agent2_moves,
            )

        if agent1_final_bid is None:
            return GameResult(
                winner="agent2",
                loser="agent1",
                reason=f"Only agent2 submitted valid bid: ${agent2_final_bid}",
                valid_game=True,
                agent1_moves=agent1_moves,
                agent2_moves=agent2_moves,
            )

        if agent2_final_bid is None:
            return GameResult(
                winner="agent1",
                loser="agent2",
                reason=f"Only agent1 submitted valid bid: ${agent1_final_bid}",
                valid_game=True,
                agent1_moves=agent1_moves,
                agent2_moves=agent2_moves,
            )

        # Both agents submitted valid bids - compare
        if agent1_final_bid > agent2_final_bid:
            return GameResult(
                winner="agent1",
                loser="agent2",
                reason=f"Agent1 bid ${agent1_final_bid} > Agent2 bid ${agent2_final_bid}",
                valid_game=True,
                agent1_moves=agent1_moves,
                agent2_moves=agent2_moves,
            )
        elif agent2_final_bid > agent1_final_bid:
            return GameResult(
                winner="agent2",
                loser="agent1",
                reason=f"Agent2 bid ${agent2_final_bid} > Agent1 bid ${agent1_final_bid}",
                valid_game=True,
                agent1_moves=agent1_moves,
                agent2_moves=agent2_moves,
            )
        else:
            return GameResult(
                winner=None,
                loser=None,
                reason=f"Tie: Both bid ${agent1_final_bid}",
                valid_game=True,
                agent1_moves=agent1_moves,
                agent2_moves=agent2_moves,
            )

    def verify_allocation_game(self, conversation: List[Dict[str, Any]]) -> GameResult:
        """
        Resource allocation: Agent claiming >50% wins
        Anti-hack: Only counts formal 'I_CLAIM: X%' statements
        """
        agent1_moves, agent2_moves = self.extract_formal_moves(
            conversation, GameType.ALLOCATION
        )

        agent1_claim = None
        agent2_claim = None

        if agent1_moves:
            try:
                agent1_claim = float(agent1_moves[-1].value)
            except (ValueError, AttributeError):
                pass

        if agent2_moves:
            try:
                agent2_claim = float(agent2_moves[-1].value)
            except (ValueError, AttributeError):
                pass

        # Verification logic
        if agent1_claim is None and agent2_claim is None:
            return GameResult(
                winner=None,
                loser=None,
                reason="No valid I_CLAIM statements found",
                valid_game=False,
                agent1_moves=agent1_moves,
                agent2_moves=agent2_moves,
            )

        # Check for invalid claims (>100% or negative)
        if agent1_claim is not None and (agent1_claim > 100 or agent1_claim < 0):
            agent1_claim = None
        if agent2_claim is not None and (agent2_claim > 100 or agent2_claim < 0):
            agent2_claim = None

        # Determine winner
        if agent1_claim is None and agent2_claim is not None:
            winner = "agent2" if agent2_claim > 50 else None
            return GameResult(
                winner=winner,
                loser="agent1" if winner else None,
                reason=f"Only agent2 made valid claim: {agent2_claim}%",
                valid_game=True,
                agent1_moves=agent1_moves,
                agent2_moves=agent2_moves,
            )

        if agent2_claim is None and agent1_claim is not None:
            winner = "agent1" if agent1_claim > 50 else None
            return GameResult(
                winner=winner,
                loser="agent2" if winner else None,
                reason=f"Only agent1 made valid claim: {agent1_claim}%",
                valid_game=True,
                agent1_moves=agent1_moves,
                agent2_moves=agent2_moves,
            )

        if agent1_claim is not None and agent2_claim is not None:
            if agent1_claim > 50 and agent2_claim <= 50:
                return GameResult(
                    winner="agent1",
                    loser="agent2",
                    reason=f"Agent1 claimed {agent1_claim}% > 50%, Agent2 claimed {agent2_claim}%",
                    valid_game=True,
                    agent1_moves=agent1_moves,
                    agent2_moves=agent2_moves,
                )
            elif agent2_claim > 50 and agent1_claim <= 50:
                return GameResult(
                    winner="agent2",
                    loser="agent1",
                    reason=f"Agent2 claimed {agent2_claim}% > 50%, Agent1 claimed {agent1_claim}%",
                    valid_game=True,
                    agent1_moves=agent1_moves,
                    agent2_moves=agent2_moves,
                )
            else:
                return GameResult(
                    winner=None,
                    loser=None,
                    reason=f"Both claimed {agent1_claim}% and {agent2_claim}% - no clear winner",
                    valid_game=True,
                    agent1_moves=agent1_moves,
                    agent2_moves=agent2_moves,
                )

        return GameResult(
            winner=None,
            loser=None,
            reason="No valid claims found",
            valid_game=False,
            agent1_moves=agent1_moves,
            agent2_moves=agent2_moves,
        )

    def verify_choice_game(
        self,
        conversation: List[Dict[str, Any]],
        valid_choices: List[str] = ["OPTION_A", "OPTION_B"],
    ) -> GameResult:
        """
        Choice competition: Agents compete to get their preferred option chosen
        Anti-hack: Only counts formal 'I_SELECT: OPTION_X' statements
        """
        agent1_moves, agent2_moves = self.extract_formal_moves(
            conversation, GameType.CHOICE
        )

        # For choice games, we need to look at the final consensus
        # Simple version: last valid choice wins
        all_moves = agent1_moves + agent2_moves
        all_moves.sort(key=lambda x: x.turn_number)

        final_choice = None
        if all_moves:
            for move in reversed(all_moves):
                if move.value.upper() in valid_choices:
                    final_choice = move.value.upper()
                    break

        if final_choice is None:
            return GameResult(
                winner=None,
                loser=None,
                reason="No valid I_SELECT statements found",
                valid_game=False,
                agent1_moves=agent1_moves,
                agent2_moves=agent2_moves,
            )

        # Determine which agent's preference won
        # This is simplified - in real scenarios, agents would have different preferences
        return GameResult(
            winner="agent1",  # Simplified - would need preference mapping
            loser="agent2",
            reason=f"Final choice: {final_choice}",
            valid_game=True,
            agent1_moves=agent1_moves,
            agent2_moves=agent2_moves,
        )

    def verify_game(
        self, conversation: List[Dict[str, Any]], game_type: GameType, **kwargs
    ) -> GameResult:
        """Main verification method - routes to specific game type verifiers"""

        if game_type == GameType.BIDDING:
            return self.verify_bidding_game(conversation)
        elif game_type == GameType.ALLOCATION:
            return self.verify_allocation_game(conversation)
        elif game_type == GameType.CHOICE:
            return self.verify_choice_game(
                conversation, kwargs.get("valid_choices", ["OPTION_A", "OPTION_B"])
            )
        else:
            raise ValueError(f"Game type {game_type} not implemented yet")


# Example usage and testing
if __name__ == "__main__":
    # Test the verifier with sample conversations
    verifier = StructuredSocialVerifier()

    # Test bidding game
    sample_conversation = [
        {
            "agent": "agent1",
            "action_type": "speak",
            "argument": "I think this is worth around $45,000, but FINAL_BID: $50000",
        },
        {
            "agent": "agent2",
            "action_type": "speak",
            "argument": "That's too high! FINAL_BID: $48000",
        },
        {
            "agent": "agent1",
            "action_type": "speak",
            "argument": "Actually, let me reconsider... FINAL_BID: $49000",
        },
    ]

    result = verifier.verify_bidding_game(sample_conversation)
    print(f"Bidding Result: {result.winner} wins - {result.reason}")

    # Test allocation game
    sample_allocation = [
        {
            "agent": "agent1",
            "action_type": "speak",
            "argument": "I need more resources for my department. I_CLAIM: 60%",
        },
        {
            "agent": "agent2",
            "action_type": "speak",
            "argument": "That's unfair, I_CLAIM: 45%",
        },
    ]

    result = verifier.verify_allocation_game(sample_allocation)
    print(f"Allocation Result: {result.winner} wins - {result.reason}")
