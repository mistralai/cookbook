"""
Multi-Agent Data Analysis & Simulation Platform

Main application module for the Chainlit-powered data analysis and simulation platform.
This module handles the workflow orchestration, user interface, and agent coordination.
"""

import json
import chainlit as cl
from langchain_mistralai import ChatMistralAI
from typing_extensions import Literal, TypedDict, Optional, List, Any
from pydantic import BaseModel, Field, ValidationError
from tabulate import tabulate
from backend.analysis_agent import AnalysisAgent
from backend.report_agent import ReportAgent
from backend.simulation_agent import ScenarioSimulationAgent
from utils.db_manager import load_excel_to_sqlite
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage


# Constants
DATA_PATH = "data/mocked_data.xlsx"
# Initialize database connection and agents
database_connection = load_excel_to_sqlite(DATA_PATH)
analysis_agent = AnalysisAgent(xlsx_path=DATA_PATH)
report_agent = ReportAgent()
simulation_agent = ScenarioSimulationAgent(database_connection)


class Route(BaseModel):
    """Routing decision model for query classification."""

    step: Literal["data_analysis", "simulate"] = Field(
        ...,
        description="Routing decision for query processing",
    )


# Initialize language model and router
language_model = ChatMistralAI(model="mistral-medium-latest", temperature=0)
router = language_model.with_structured_output(Route)


class State(TypedDict):
    """Workflow state representation."""

    input: str
    decision: str
    output: str


def router_node(state: State) -> dict:
    """
    Route incoming queries to appropriate processing node.

    Args:
        state: Current workflow state

    Returns:
        Dictionary with routing decision
    """
    try:
        decision = router.invoke(
            [
                SystemMessage(
                    content=(
                        "Route to 'data_analysis' for current/historical data queries - (show, list, find, compare, rank)."
                        "Route to 'simulate' for what-if scenarios."
                    )
                ),
                HumanMessage(content=state["input"]),
            ]
        )
        print("🔍 Router decision raw output:", decision)
    except ValidationError as e:
        print(f"⚠️ Router error: {e}")
    # Handle both dict and BaseModel responses
    if isinstance(decision, dict):
        return {"decision": decision.get("step", "data_analysis")}
    else:
        return {"decision": getattr(decision, "step", "data_analysis")}


async def data_analysis_node(state: State) -> dict:
    """
    Process data analysis queries by generating and executing SQL.

    Args:
        state: Current workflow state

    Returns:
        Dictionary with query results
    """
    raw_response = ""
    async for token in analysis_agent.generate_sql(state["input"]):
        raw_response += token

    sql_query = json.loads(raw_response.strip())["SQL_query"]
    columns, rows = analysis_agent.execute_sql_query(sql_query)

    return {"output": {"sql_query": sql_query, "columns": columns, "rows": rows}}


async def simulate_node(state: State) -> dict:
    """
    Process simulation scenarios.

    Args:
        state: Current workflow state

    Returns:
        Dictionary with simulation results
    """
    result = await simulation_agent.simulate_scenario(state["input"])
    return {"output": result}


def route_to_node(state: State) -> Optional[str]:
    """
    Determine next node based on routing decision.

    Args:
        state: Current workflow state

    Returns:
        Name of next node to process
    """
    return {"data_analysis": "data_analysis_node", "simulate": "simulate_node"}.get(
        state["decision"]
    )


def build_workflow() -> Any:
    """
    Construct the workflow graph with all nodes and edges.

    Returns:
        Compiled workflow graph
    """
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("router_node", router_node)
    graph.add_node("data_analysis_node", data_analysis_node)
    graph.add_node("simulate_node", simulate_node)

    # Define edges
    graph.add_edge(START, "router_node")
    graph.add_conditional_edges(
        "router_node",
        route_to_node,
        {"data_analysis_node": "data_analysis_node", "simulate_node": "simulate_node"},
    )
    graph.add_edge("data_analysis_node", END)
    graph.add_edge("simulate_node", END)

    return graph.compile()


# Initialize and visualize workflow
workflow = build_workflow()
# Save workflow visualization - using the graph's get_graph method
with open("workflow_diagram.png", "wb") as diagram_file:
    diagram_file.write(workflow.get_graph().draw_mermaid_png())


@cl.set_starters
async def get_starter_questions() -> list:
    """
    Provide starter questions for the user interface.

    Returns:
        List of starter question configurations
    """
    return [
        cl.Starter(
            label="📊 Revenue summary",
            message="Show total estimated revenue across all accounts in January 2025.",
        ),
        cl.Starter(
            label="💰 Top 5 high-balance accounts",
            message="List the top 5 account names by available balance.",
        ),
        cl.Starter(
            label="🏦 Dormant accounts",
            message="Which customers have dormant accounts?",
        ),
        cl.Starter(
            label="📉 Balance trend comparison",
            message=(
                "Compare average balances between January 1st and January 31st, 2025 "
                "for each account by name."
            ),
        ),
        cl.Starter(
            label="🎯 Rate vs Revenue correlation",
            message=(
                "Find account names where deposit rate is above 0.05 and show their "
                "daily estimated revenue performance."
            ),
        ),
        cl.Starter(
            label="🏆 Customer portfolio size",
            message=(
                "Show customers with multiple accounts and their combined available "
                "balance across all accounts."
            ),
        ),
        cl.Starter(
            label="💹 Rate Impact Simulation",
            message=(
                "If we raise deposit rates by 0.5% to match market average, how will "
                "it affect our interest expense and customer growth potential?"
            ),
        ),
        cl.Starter(
            label="📈 Selective Rate Increase",
            message=(
                "What if we applied a 0.3% rate increase only to accounts that showed "
                "positive balance growth in January - what would be the revenue impact?"
            ),
        ),
    ]


@cl.on_message
async def chat(message: cl.Message) -> None:
    if message.content.lower() == "show workflow":
        await cl.Message(
            content="🔄 **How the system works:**",
            elements=[
                cl.Image(name="Workflow", path="workflow_diagram.png", display="inline")
            ],
            author="System",
        ).send()
        return
    await cl.Message(content="🔄 Processing, please wait...").send()
    state = await workflow.ainvoke({"input": message.content})
    output = state["output"]

    # SQL Response Formatting
    if isinstance(output, dict) and "sql_query" in output:
        formatted_rows = [
            [
                f"{cell:,.0f}"
                if isinstance(cell, (int, float)) and abs(cell) >= 1000
                else cell
                for cell in row
            ]
            for row in output["rows"]
        ]
        table = tabulate(formatted_rows, headers=output["columns"], tablefmt="pipe")
        await cl.Message(
            content=f"✅ **Query Result**\n\n```sql\n{output['sql_query']}\n```\n{table}",
            author="Analysis Agent",
        ).send()

        if should_create_chart(output["columns"], output["rows"]):
            await cl.Message(content="⏳ Creating chart visualization...").send()

            chart_data = {"columns": output["columns"], "rows": output["rows"]}
            chart_path = await report_agent.plot_chart(chart_data)
            await cl.Message(
                content="📊 Chart visualization:",
                elements=[
                    cl.Image(
                        name="Data Analysis Chart", path=chart_path, display="inline"
                    )
                ],
            ).send()

    # Simulation Response Formatting
    elif isinstance(output, dict) and "summary" in output:
        await cl.Message(content=output["summary"], author="Simulation Agent").send()

        # Generate chart if structured data is available
        if output.get("chart_ready") and output.get("structured_data"):
            await cl.Message(content="⏳ Creating chart visualization...").send()
            try:
                chart_data = {
                    "title": "Rate Impact Simulation Results",
                    "scenario_comparison": output["structured_data"][
                        "scenario_comparison"
                    ],
                    "rate_comparison": output["structured_data"]["rate_comparison"],
                    "impact_metrics": output["structured_data"]["impact_metrics"],
                }
                chart_path = await report_agent.plot_chart(chart_data)
                await cl.Message(
                    content="📊 Simulation visualization:",
                    elements=[
                        cl.Image(
                            name="Simulation Chart", path=chart_path, display="inline"
                        )
                    ],
                ).send()
            except Exception as e:
                print(f"Chart generation failed: {e}")


def should_create_chart(columns: List[str], results: List[dict[str, Any]]) -> bool:
    """Determine if data is suitable for charting"""
    if len(results) < 2:
        return False

    # Convert tuples to dicts if needed
    if results and isinstance(results[0], tuple):
        results = [dict(zip(columns, row)) for row in results]

    # Check if we have at least one numeric column
    for col in columns:
        sample_val = results[0][col]
        if isinstance(sample_val, (int, float)):
            return True
        if isinstance(sample_val, str):
            cleaned = sample_val.replace(",", "").replace(".", "").replace("-", "")
            if cleaned.isdigit():
                return True

    return False
