"""Basic example of using Sotopia Verifiable."""

from sotopia_verifiable import VerifiableGame

# Create a game configuration
game_config = {
    "name": "Simple Verification Example",
    "verification_rules": [
        {"type": "score_threshold", "min_score": 0.7}
    ]
}

# Create a verifiable game
game = VerifiableGame(game_config)

# Add another verification rule
game.add_verification_rule({"type": "time_limit", "max_seconds": 300})

# Print the verification rules
print("Verification Rules:")
for i, rule in enumerate(game.get_verification_rules(), 1):
    print(f"Rule {i}: {rule}")

# Simulate a game outcome
game_outcome = {
    "score": 0.85,
    "time_taken": 250
}

# Verify the outcome
result = game.verify(game_outcome)

# Print the verification result
print("\nVerification Result:")
print(f"Valid: {result.is_valid}")
print(f"Score: {result.score}")
print(f"Details: {result.details}")

# Note: The base implementation always returns valid=True
# Custom implementations would check against the actual rules