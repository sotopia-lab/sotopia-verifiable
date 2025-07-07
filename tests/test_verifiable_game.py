"""Tests for the VerifiableGame class."""

import pytest

from sotopia_verifiable.core.verifiable_game import VerifiableGame, VerificationResult


def test_verifiable_game_initialization():
    """Test that a VerifiableGame can be initialized."""
    game_config = {"name": "Test Game", "verification_rules": []}
    game = VerifiableGame(game_config)
    
    assert game.game_config == game_config
    assert game.verification_rules == []


def test_verifiable_game_add_rule():
    """Test that verification rules can be added to a game."""
    game = VerifiableGame({"name": "Test Game"})
    
    rule = {"type": "score_threshold", "min_score": 0.7}
    game.add_verification_rule(rule)
    
    assert game.verification_rules == [rule]
    
    # Add another rule
    rule2 = {"type": "time_limit", "max_seconds": 300}
    game.add_verification_rule(rule2)
    
    assert game.verification_rules == [rule, rule2]


def test_verifiable_game_verify():
    """Test the basic verification functionality."""
    game = VerifiableGame({"name": "Test Game"})
    
    # The base implementation should always return a valid result
    result = game.verify({})
    
    assert isinstance(result, VerificationResult)
    assert result.is_valid is True
    assert result.score == 1.0
    assert "message" in result.details