# sotopia-verifiable
A collection of verifiable games in sotopia format

- scenario_generator.py
  * Defines social‐interaction "axes" as code->label mappings
  * Samples random combinations of those axes
  * Prompts the OpenAI API to produce JSON‑structured scenarios
  * Inserts each scenario (with UUID and axis codes) into SQLite via DBHelper

- scenario_runner.py
  * Takes a scenario UUID as CLI argument
  * Loads that row from SQLite and parses the richer agents_json field
  * Ensures AgentProfile entries (with full profile fields) and an EnvironmentProfile in Redis
  * Builds a UniformSampler and invokes run_async_server to execute the scenario as a multi‐agent game
