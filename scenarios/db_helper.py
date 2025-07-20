# db_helper.py

import sqlite3
import json
from typing import Dict, Any


class DBHelper:
    def __init__(self, db_path: str = "scenarios.db") -> None:
        # Connect to the SQLite database (creates file if it doesn't exist)
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self) -> None:
        c = self.conn.cursor()
        # Create scenarios table with only the currently used axes
        c.execute("""
        CREATE TABLE IF NOT EXISTS scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uuid TEXT UNIQUE,
            title TEXT,
            description TEXT,
            agents_json TEXT,
            win_condition TEXT,
            interdependence TEXT,
            relational_model TEXT,
            resource_type TEXT,
            context TEXT,
            evaluation TEXT,
            agent_count INTEGER,
            verifiability TEXT
        )
        """)
        self.conn.commit()

    def reset_db(self) -> None:
        """Drops and recreates the scenarios table."""
        c = self.conn.cursor()
        c.execute("DROP TABLE IF EXISTS scenarios")
        self.conn.commit()
        self._create_table()

    def insert_scenario(
        self, scenario_json: Dict[str, Any], axes: Dict[str, Any]
    ) -> None:
        """
        Inserts a scenario into the database.
        scenario_json: dict with keys uuid, title, description, agents, win_condition
        axes: dict of axis_name -> axis_value (only current axes)
        """
        c = self.conn.cursor()
        c.execute(
            """
        INSERT OR REPLACE INTO scenarios (
            uuid,
            title,
            description,
            agents_json,
            win_condition,
            interdependence,
            relational_model,
            resource_type,
            context,
            evaluation,
            agent_count,
            verifiability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                scenario_json["uuid"],
                scenario_json["title"],
                scenario_json["description"],
                json.dumps(scenario_json["agents"], ensure_ascii=False),
                scenario_json["win_condition"],
                axes["interdependence"],
                axes["relational_model"],
                axes["resource_type"],
                axes["context"],
                axes["evaluation"],
                axes["agent_count"],
                axes["verifiability"],
            ),
        )
        self.conn.commit()

    def close(self) -> None:
        """Closes the database connection."""
        self.conn.close()
