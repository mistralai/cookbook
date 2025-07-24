"""
Simulation Agent Module

This module contains the ScenarioSimulationAgent class which handles what-if scenario
simulations for banking data analysis.
"""

import os
import json
import pandas as pd
import sqlite3
from mistralai import Mistral

# Canadian market rates (realistic for RBC demo)
MARKET_RATES = {
    "deposit_rate": 4.25,  # 4.25% - Canadian avg savings rate
    "overdraft_rate": 19.5,  # 19.5% - Canadian avg overdraft rate
    "prime_rate": 6.45,  # 6.45% - Bank of Canada rate context
    "mortgage_rate": 5.25,  # 5.25% - Canadian mortgage benchmark
}

SIMULATION_PROMPT = f"""
You are a banking simulation agent for RBC (Royal Bank of Canada). You have access to account data and market intelligence.

MARKET CONTEXT (Canadian Banking):
- Average deposit rate: {MARKET_RATES["deposit_rate"]}%
- Average overdraft rate: {MARKET_RATES["overdraft_rate"]}%
- Prime rate: {MARKET_RATES["prime_rate"]}%

CALCULATION RULES:
1. For positive balances: Daily cost = balance × deposit_rate ÷ 365
2. For negative balances: Daily revenue = |balance| × overdraft_rate ÷ 365
3. Net daily revenue = overdraft revenue - deposit costs

RESPONSE FORMAT:
Return JSON with:
{{
  "summary": "Brief strategic analysis with financial impact",
  "structured_data": {{
    "scenario_comparison": [
      {{"metric": "Daily Revenue", "current": X, "simulated": Y}},
      {{"metric": "Deposit Cost", "current": X, "simulated": Y}},
      {{"metric": "Overdraft Revenue", "current": X, "simulated": Y}}
    ],
    "rate_comparison": [
      {{"rate_type": "Deposit Rate", "current": X, "market": {MARKET_RATES["deposit_rate"]}, "simulated": Y}},
      {{"rate_type": "Overdraft Rate", "current": X, "market": {MARKET_RATES["overdraft_rate"]}, "simulated": Y}}
    ],
    "impact_metrics": {{
      "daily_impact": X,
      "monthly_impact": X,
      "accounts_affected": X
    }}
  }},
  "chart_ready": true
}}

Provide strategic insights on competitive positioning, customer impact, and revenue implications.
"""


class ScenarioSimulationAgent:
    """Agent for simulating banking rate scenarios."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """
        Initialize the simulation agent.

        Args:
            conn: SQLite database connection
        """
        self.conn = conn  # Use existing connection from db_manager
        self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

        self.agent = self.client.beta.agents.create(
            name="simulation_agent",
            model="mistral-large-latest",
            instructions=SIMULATION_PROMPT,
            description="Simulates banking rate scenarios with strategic insights.",
            completion_args={"response_format": {"type": "json_object"}},
        )

    def get_accounts_df(self) -> pd.DataFrame:
        """Get accounts data from accounts table."""
        query = """SELECT account_number, customer_name, available_balance,
                   deposit_rate, overdraft_rate FROM accounts"""
        return pd.read_sql_query(query, self.conn)

    def get_daily_metrics_df(self) -> pd.DataFrame:
        """Get daily metrics data from daily_metrics table."""
        query = """SELECT account_number, date, available_balance, estimated_revenue,
                   net_change FROM daily_metrics ORDER BY account_number, date"""
        return pd.read_sql_query(query, self.conn)

    async def simulate_scenario(self, user_query: str) -> dict:
        """
        Run simulation using Mistral agent.

        Args:
            user_query: The scenario to simulate

        Returns:
            Dictionary containing simulation results
        """
        df = self.get_accounts_df()
        daily_df = self.get_daily_metrics_df()

        # Prepare data for agent
        account_data = df.to_dict(orient="records")
        daily_data = daily_df.to_dict(orient="records")

        # Calculate current portfolio summary
        positive_bal = df[df["available_balance"] >= 0]
        negative_bal = df[df["available_balance"] < 0]

        portfolio_summary = {
            "total_deposits": float(positive_bal["available_balance"].sum()),
            "total_overdrafts": float(abs(negative_bal["available_balance"].sum())),
            "avg_deposit_rate": float(positive_bal["deposit_rate"].mean()) * 100,
            "avg_overdraft_rate": float(negative_bal["overdraft_rate"].mean()) * 100,
            "accounts_count": len(df),
        }

        # Create conversation with agent
        response = self.client.beta.conversations.start(
            agent_id=self.agent.id,
            inputs=f"""
            User Query: {user_query}

            Portfolio Summary:
            - Total Deposits: ${portfolio_summary["total_deposits"]:,.2f}
            - Total Overdrafts: ${portfolio_summary["total_overdrafts"]:,.2f}
            - Average Deposit Rate: {portfolio_summary["avg_deposit_rate"]:.2f}%
            - Accounts Count: {portfolio_summary["accounts_count"]}

            Sample Account Data: {json.dumps(account_data[:5], indent=2)}
            Sample Daily Metrics: {json.dumps(daily_data[:5], indent=2)}
            """,
        )

        # Parse response
        try:
            # Find the first MessageOutputEntry in the outputs
            message_output = None
            for output in response.outputs:
                if hasattr(output, "content"):
                    message_output = output
                    break

            if message_output is None:
                raise ValueError("No valid message output found")

            output_content = message_output.content
            # Handle both string and chunk list cases
            if isinstance(output_content, list):
                # If it's a list of chunks, extract the text content
                json_str = "".join(
                    chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                    for chunk in output_content
                )
            else:
                json_str = output_content

            result = json.loads(json_str)
            return result
        except (json.JSONDecodeError, AttributeError, ValueError):
            # Fallback response
            return {
                "summary": f"Simulation analysis for: {user_query}",
                "structured_data": {
                    "scenario_comparison": [],
                    "rate_comparison": [],
                    "impact_metrics": {
                        "daily_impact": 0,
                        "monthly_impact": 0,
                        "accounts_affected": len(df),
                    },
                },
                "chart_ready": False,
            }
