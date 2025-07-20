# scenario_generator.py
import itertools
import uuid
import openai
import json
import random
import os
from db_helper import DBHelper
from typing import Dict, Any

# 1. Define design axes as a mapping
axes = {
    "interdependence": {
        "cooperative": "cooperative (positive interdependence)",
        "competitive": "competitive (negative interdependence)",
        "individualistic": "individualistic (no interdependence)",
    },
    "relational_model": {
        "communal_sharing": "communal sharing (people conceived as equivalent, undifferentiated, interchangeable)",
        "authority_ranking": "authority ranking (hierarchical roles)",
        "equality_matching": "equality matching (one‐for‐one correspondence)",
        "market_pricing": "market pricing (norms of proportionality and equity)",
    },
    "resource_type": {
        "money": "money",
        "information": "information",
        "goods": "goods",
        "services": "services",
        "status": "status",
        "love": "love",
    },
    "context": {
        "office": "office",
        "market": "market",
        "online_forum": "online forum",
        "family_dinner": "family dinner",
        "emergency_scene": "emergency scene",
        "workplace": "workplace",
        "school": "school",
        "public_space": "public space",
        "virtual_community": "virtual community",
    },
    "evaluation": {
        "binary_win_loss": "binary win/loss",
    },
    "agent_count": {
        "2": "2",  # extend to multiple agents later on
    },
    "verifiability": {
        "explicit_rules": "explicit rules",
    },
}

# 2. Build all combinations over the *keys* of each axis
axis_keys = list(axes.keys())
axis_values = [list(axes[k].keys()) for k in axis_keys]
random.seed(42)
combos = random.sample(list(itertools.product(*axis_values)), 10)

# 3. Initialize DB and OpenAI
db = DBHelper("scenarios.db")
db.reset_db()  # Clear existing scenarios
openai.api_key = os.getenv("OPENAI_API_KEY")

# 4. Prompt template enforcing JSON output
template = (
    # "Design a {stage}, text-only interaction with {agent_count} agents in a {interdependence} interdependence setting, "
    "Design a text-only interaction with {agent_count} agents in a {interdependence} setting, "
    # "following {relational_model} norms, negotiating over {resource_type} under a {interaction_mode} framework, "
    "following {relational_model} norms, negotiating over {resource_type}, "
    "set in a {context} context, with {evaluation} outcome criteria that is {verifiability} verifiable.\n\n"
    "Output ONLY a JSON object with keys:\n"
    "title (string), description (string), agents (list of dicts with keys: first_name, last_name, age, occupation, gender, decision_making_style, secret), win_condition (string)."
)

# 5. Generate and store scenarios
for combo in combos:
    # combo is a tuple of *code* values, e.g. ('cooperative','communal_sharing',…)
    code_params = dict(zip(axis_keys, combo))
    # map to human labels
    prompt_params = {k: axes[k][v] for k, v in code_params.items()}
    prompt = template.format(**prompt_params)

    resp = openai.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": "You are a scenario generator."},
            {"role": "user", "content": prompt},
        ],
    )

    content = resp.choices[0].message.content
    assert isinstance(content, str)

    try:
        scenario: Dict[str, Any] = json.loads(content)
        scenario.setdefault("uuid", str(uuid.uuid4()))
        db.insert_scenario(scenario, code_params)
        print("Inserted", scenario["uuid"])
    except Exception as e:
        print("Failed for", code_params, e)

db.close()
