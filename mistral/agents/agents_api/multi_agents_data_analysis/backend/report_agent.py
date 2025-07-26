#!/usr/bin/env python
"""
Report Agent Module

This module contains the ReportAgent class which handles chart generation
and visualization of data analysis results.
"""

import os
import json
from typing import Optional, Any
from pathlib import Path
from mistralai import Mistral

MODEL = "mistral-medium-latest"


class ReportAgent:
    """Agent for generating visualizations from data."""

    def __init__(self) -> None:
        """Initialize the Mistral client and agent."""
        api_key = os.environ["MISTRAL_API_KEY"]
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable not set.")
        self.client = Mistral(api_key)
        self.agent = self.client.beta.agents.create(
            model=MODEL,
            name="report_agent",
            instructions=(
                "You are a data visualization agent that always creates charts using the code interpreter tool. "
                "When provided with data (especially in tabular format with 'columns' and 'rows'), you must immediately generate a suitable chart. "
                "Never ask the user for clarification or confirmation. Do not explain what you will do — just do it. "
                "Use appropriate chart types (e.g. bar, line, pie) based on the data. "
                "Always format large numbers with commas and avoid scientific notation. "
                "Label axes and titles clearly. ALWAYS Use a logarithmic scale if necessary. "
                "Return the chart as an image using the tool_file output. "
                "Do not include any conversational text unless required to explain the chart."
            ),
            description="Creates charts using the provided data.",
            tools=[{"type": "code_interpreter"}],
        )

    async def plot_chart(self, data: Any) -> Optional[str]:
        """
        Generate a chart from the provided data.

        Args:
            data: The data to visualize

        Returns:
            Path to the generated chart file or None if chart generation failed
        """

        if not isinstance(data, list):
            data = [data]

        data_str = json.dumps(data, indent=2)
        print(f"Message: {data_str}")

        response = self.client.beta.conversations.start(
            agent_id=self.agent.id,
            inputs=data_str,
        )

        print(f"Response: {response}")
        for output in response.outputs:
            if hasattr(output, "content"):
                content = output.content
                if isinstance(content, list):
                    for chunk in content:
                        # Use attribute access — chunk is likely a ToolFileChunk object
                        if (
                            hasattr(chunk, "file_id")
                            and getattr(chunk, "type", None) == "tool_file"
                        ):
                            file_name = getattr(chunk, "file_name", "chart.png")
                            chart_path = Path("charts") / f"chart_{file_name}"
                            chart_path.parent.mkdir(exist_ok=True)

                            file_content = self.client.files.download(
                                file_id=chunk.file_id
                            ).read()
                            chart_path.write_bytes(file_content)

                            return str(chart_path)

        return None  # No chart found
