from serpapi import GoogleSearch
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv
import json
from datetime import datetime
from loguru import logger

load_dotenv()


class HotelSearchError(Exception):
    """Custom exception for hotel search errors"""

    pass


class HotelSearchParams:
    """Class to validate and format hotel search parameters"""

    def __init__(self, **kwargs):
        self.query = kwargs.get("query", "")
        self.check_in_date = kwargs.get("check_in_date", "")
        self.check_out_date = kwargs.get("check_out_date", "")
        self.adults = kwargs.get("adults", "2")
        self.max_price = kwargs.get("max_price", "1000")
        self.min_price = kwargs.get("min_price", "100")
        self.rating = kwargs.get("rating", "8")
        self.hotel_class = kwargs.get("hotel_class", "2")
        self.sort_by = kwargs.get("sort_by", "3")

        self._validate_dates()
        self._validate_numeric_params()

    def _validate_dates(self):
        try:
            if self.check_in_date:
                datetime.strptime(self.check_in_date, "%Y-%m-%d")
            if self.check_out_date:
                datetime.strptime(self.check_out_date, "%Y-%m-%d")
        except ValueError:
            raise HotelSearchError("Invalid date format. Please use YYYY-MM-DD")

    def _validate_numeric_params(self):
        try:
            self.adults = str(max(1, min(10, int(self.adults))))
            self.max_price = str(max(0, int(self.max_price)))
            self.min_price = str(max(0, min(int(self.max_price), int(self.min_price))))
            self.rating = str(max(7, min(9, int(self.rating))))
            self.hotel_class = str(max(2, min(4, int(self.hotel_class))))
            self.sort_by = str(max(3, min(13, int(self.sort_by))))
        except (ValueError, TypeError):
            raise HotelSearchError("Invalid numeric parameters")

    def to_dict(self) -> Dict:
        return {
            "engine": "google_hotels",
            "q": self.query,
            "check_in_date": self.check_in_date,
            "check_out_date": self.check_out_date,
            "adults": self.adults,
            "currency": "USD",
            "sort_by": self.sort_by,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "rating": self.rating,
            "hotel_class": self.hotel_class,
            "gl": "us",
            "hl": "en",
            "api_key": os.environ.get("SERPAPI_KEY"),
        }


class HotelSearchResult:
    """Class to format and validate hotel search results"""

    def __init__(self, results: Dict, maximum_results: int = 3):
        self.results = results
        self.maximum_results = maximum_results
        self._validate_results()

    def _validate_results(self):
        if not isinstance(self.results, dict):
            raise HotelSearchError("Invalid results format")
        if "properties" not in self.results:
            raise HotelSearchError("No properties found in results")

    def format(self) -> str:
        data = []
        for property_item in self.results["properties"][: self.maximum_results]:
            try:
                hotel_data = {
                    "name": property_item.get("name", "Unknown"),
                    "description": property_item.get("description", ""),
                    "type": property_item.get("type", "Hotel"),
                    "price": property_item.get("rate_per_night", {}).get(
                        "lowest", "N/A"
                    ),
                    "nearby_places": property_item.get("nearby_places", ""),
                    "rating": property_item.get("overall_rating", "N/A"),
                    "reviews": property_item.get("reviews", "N/A"),
                    "url": property_item.get("link", ""),
                    "amenities": ", ".join(property_item.get("amenities", [])),
                }
                data.append(hotel_data)
            except Exception as e:
                logger.error(f"Error formatting hotel data: {e}")
                continue

        return json.dumps({"hotels": data}, indent=2)


SEARCH_HOTEL_TOOL = {
    "type": "function",
    "function": {
        "name": "search_hotel_serpapi",
        "description": "Search hotel based on user preferences",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to search for. Example: 'hotel in Paris' or 'apartments in London United Kingdom'",
                },
                "check_in_date": {
                    "type": "string",
                    "description": "Date of arrival in the following format YYYY-MM-DD.",
                },
                "check_out_date": {
                    "type": "string",
                    "description": "Date of departure in the following format YYYY-MM-DD.",
                },
                "adults": {
                    "type": "string",
                    "description": "The number of adults staying (1-10).",
                },
                "max_price": {
                    "type": "string",
                    "description": "Optional, the maximum price per night in USD.",
                },
                "min_price": {
                    "type": "string",
                    "description": "Optional, the minimum price per night in USD.",
                },
                "rating": {
                    "type": "string",
                    "description": "Optional, the minimum rating of the hotel (7-9). 7 is 3.5+ stars, 8 is 4+ stars, 9 is 4.5+ stars",
                },
                "hotel_class": {
                    "type": "string",
                    "description": "Optional, the minimum class of the hotel (2-4). 2 is 3 stars, 3 is 4 stars, 4 is 5 stars",
                },
                "sort_by": {
                    "type": "string",
                    "description": "Optional, the sorting order of the hotels (3-13). 3 is lowest price, 8 is highest rating, 13 is most reviewed",
                },
            },
            "required": ["query", "check_in_date", "check_out_date", "adults"],
        },
    },
}


def search_hotel_serpapi(**kwargs) -> str:
    """
    Search for hotels using the SerpAPI Google Hotels search engine.

    Args:
        **kwargs: Hotel search parameters

    Returns:
        str: JSON string containing formatted hotel search results

    Raises:
        HotelSearchError: If there are validation errors or API errors
    """
    try:
        # Validate and format parameters
        params = HotelSearchParams(**kwargs)

        # Perform search
        search = GoogleSearch(params.to_dict())
        results = search.get_dict()

        # Format results
        return HotelSearchResult(results).format()

    except Exception as e:
        logger.error(f"Hotel search error: {e}")
        raise HotelSearchError(f"Failed to search hotels: {str(e)}")
