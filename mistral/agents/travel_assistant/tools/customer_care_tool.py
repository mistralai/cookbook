from serpapi import GoogleSearch
from typing import Optional
import os
from dotenv import load_dotenv
import json

load_dotenv()

MODIFY_BOOKING_TOOL = {
    "type": "function",
    "function": {
        "name": "modify_booking",
        "description": "Modify booking of a customer",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The action to perform. Can be 'modify' or 'cancel'.",
                },
                "reference_number": {
                    "type": "string",
                    "description": "The reference number for the booking",
                },
                "modification_details": {
                    "type": "string",
                    "description": "The details of the modification (eg: checkin date, checkout date, number of adults, number of children, etc.)",
                },
            },
            "required": ["action", "modification_details"],
        },
    },
}


def modify_booking(
    action: str, modification_details: str, reference_number: Optional[str] = "54388383"
):
    """
    Modify booking of a customer
    """
    var = ""
    if reference_number:
        var = f" with reference number {reference_number}"

    final_message = f" Action: {action} {var}"
    fake_response = {
        "status": "success 🎉",
        "message": final_message,
    }
    return json.dumps(fake_response)
