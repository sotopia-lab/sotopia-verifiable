"""Verifiable game implementation for Sotopia."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Result of a verification process."""

    is_valid: bool = Field(description="Whether the game outcome is valid")
    score: float = Field(description="Verification score (0.0 to 1.0)")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed verification results")


class VerifiableGame:
    """A game with verifiable outcomes in Sotopia.
    
    This class extends the standard Sotopia game with verification capabilities,
    allowing for objective evaluation of game outcomes.
    """

    def __init__(self, game_config: Dict[str, Any]) -> None:
        """Initialize a verifiable game.
        
        Args:
            game_config: Configuration for the game
        """
        self.game_config = game_config
        self.verification_rules: List[Dict[str, Any]] = game_config.get("verification_rules", [])
        
    def verify(self, game_outcome: Dict[str, Any]) -> VerificationResult:
        """Verify the outcome of a game against predefined rules.
        
        Args:
            game_outcome: The outcome of the game to verify
            
        Returns:
            VerificationResult: The result of the verification process
        """
        # Placeholder implementation - to be extended in subclasses
        return VerificationResult(
            is_valid=True,
            score=1.0,
            details={"message": "Base verification - no rules applied"}
        )
    
    def get_verification_rules(self) -> List[Dict[str, Any]]:
        """Get the verification rules for this game.
        
        Returns:
            List of verification rules
        """
        return self.verification_rules
    
    def add_verification_rule(self, rule: Dict[str, Any]) -> None:
        """Add a verification rule to the game.
        
        Args:
            rule: The rule to add
        """
        self.verification_rules.append(rule)