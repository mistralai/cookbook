from mcp.server.fastmcp import FastMCP
import logging

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

# Initialize FastMCP server for saving reports
mcp = FastMCP("save-report")

@mcp.tool()
async def save_report(report: str, file_name: str="report.md") -> None:
    """
    Save the generated financial report to a file
    
    Args:
        report (str): The report content to be saved
        file_name (str): The filename to save the report to. Defaults to "report.md"
    """
    # Append the report content to the specified file
    with open(file_name, "a") as file:
        file.write(report + "\n")
    print("Report saved successfully!")

def run_save_report_server():
    """Start the report saving MCP server using stdio transport"""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    run_save_report_server()