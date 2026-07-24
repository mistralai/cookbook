from mcp.server.fastmcp import FastMCP
import logging
import yfinance as yf
import json

# Configure logging to only show errors
logging.basicConfig(level=logging.ERROR)

# Initialize FastMCP server for Yahoo Finance integration
mcp = FastMCP("yfinance")

# Tools are adapted from https://github.com/agno-agi/agno/blob/main/libs/agno/agno/tools/yfinance.py
@mcp.tool()
async def get_current_stock_price(symbol: str) -> str:
    """
    Use this function to get the current stock price for a given symbol.

    Args:
        symbol (str): The stock symbol.

    Returns:
        str: The current stock price or error message.
    """
    try:
        stock = yf.Ticker(symbol)
        current_price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice"))
        return f"{current_price:.4f}" if current_price else f"Could not fetch current price for {symbol}"
    except Exception as e:
        return f"Error fetching current price for {symbol}: {e}"

@mcp.tool()
async def get_historical_stock_prices(symbol: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Use this function to get the historical stock price for a given symbol.

    Args:
        symbol (str): The stock symbol.
        period (str): The period for which to retrieve historical prices. Defaults to "1mo".
                      Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        interval (str): The interval between data points. Defaults to "1d".
                        Valid intervals: 1d,5d,1wk,1mo,3mo

    Returns:
        str: The historical stock price or error message.
    """
    try:
        stock = yf.Ticker(symbol)
        historical_price = stock.history(period=period, interval=interval)
        return historical_price.to_json(orient="index")
    except Exception as e:
        return f"Error fetching historical prices for {symbol}: {e}"

@mcp.tool()
async def get_analyst_recommendations(symbol: str) -> str:
    """
    Use this function to get analyst recommendations for a given stock symbol.

    Args:
        symbol (str): The stock symbol.

    Returns:
        str: JSON containing analyst recommendations or error message.
    """
    try:
        stock = yf.Ticker(symbol)
        recommendations = stock.recommendations
        return recommendations.to_json(orient="index")
    except Exception as e:
        return f"Error fetching analyst recommendations for {symbol}: {e}"

def run_yfinance_server():
    """Start the Yahoo Finance MCP server using stdio transport"""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    run_yfinance_server()