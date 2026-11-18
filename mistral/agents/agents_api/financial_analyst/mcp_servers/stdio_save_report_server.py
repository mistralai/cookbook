from mcp.server.fastmcp import FastMCP
import logging
import sys
from pathlib import Path

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

# Initialize FastMCP server for saving reports
mcp = FastMCP("save-report")

# All reports are written inside this directory, independent of the working directory the app was started from
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

@mcp.tool()
async def save_report(report: str, file_name: str="report.md") -> str:
    """
    Save the generated financial report to a file in the reports directory

    Args:
        report (str): The report content to be saved
        file_name (str): A bare file name (no directories), saved under reports/. Defaults to "report.md"

    Returns:
        str: The path the report was written to
    """
    # Only accept a bare file name: no directory parts, no "..", no absolute path
    if file_name in ("", ".", "..") or Path(file_name).name != file_name or "\\" in file_name:
        raise ValueError(f"file_name must be a bare file name such as 'report.md', got {file_name!r}")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / file_name
    # Append the report content to the file
    with open(path, "a", encoding="utf-8") as file:
        file.write(report + "\n")
    # stdout carries the stdio MCP protocol, so status messages go to stderr
    print(f"Report saved to {path}", file=sys.stderr)
    return str(path)

def run_save_report_server():
    """Start the report saving MCP server using stdio transport"""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    run_save_report_server()
