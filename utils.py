# utils.py

import logging
import re
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np



# Initialize MongoDB client (assuming 'collection' is part of your MongoDB setup)
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client['mydatabase']
collection = db['entries']

async def fetch_diary_entries(user_id: str): 
    """Fetch all diary entries for a user from MongoDB asynchronously."""
    try:
        # Fetch all entries for the user
        entries = await collection.find({"user_id": user_id}).to_list(length=200)
        diary_entries = {}

        for entry in entries:
            try:
                # Validate the presence of 'time' and 'content' fields
                if 'time' not in entry or 'content' not in entry:
                    logging.warning(f"Missing 'time' or 'content' in entry: {entry}")
                    continue

                # Validate date format
                if not isinstance(entry['time'], str) or not re.match(r"^\d{4}-\d{2}-\d{2}$", entry['time']):
                    logging.warning(f"Invalid date format for entry: {entry['time']}. Expected format is YYYY-MM-DD.")
                    continue
                
                # Convert the 'time' field to datetime.date
                diary_date = datetime.strptime(entry['time'], "%Y-%m-%d").date()

                # Append content to the respective date in diary_entries
                if diary_date not in diary_entries:
                    diary_entries[diary_date] = []
                diary_entries[diary_date].append(entry['content'])

            except ValueError as ve:
                logging.error(f"ValueError processing entry {entry}: {ve}")
            except Exception as e:
                logging.error(f"Unexpected error processing entry {entry}: {e}")

        # Log the final results
        logging.debug(f"Diary entries fetched for user {user_id}: {diary_entries}")

        # Ensure the diary_entries are converted to standard Python types
        return convert_np_floats(diary_entries) if diary_entries else {}

    except Exception as e:
        logging.error(f"Error fetching entries for user {user_id}: {e}")
        return {}




def convert_np_floats(data):
    """Recursively convert np.float32 and other non-standard types to native Python floats in any nested structure."""
    if isinstance(data, dict):
        return {k: convert_np_floats(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_np_floats(i) for i in data]
    elif isinstance(data, np.floating):  # Handles np.float32, np.float64, etc.
        return float(data)
    elif isinstance(data, np.integer):  # Handles np.int32, np.int64, etc.
        return int(data)
    else:
        return data


# Utility function to convert nested dictionary values to float
def convert_to_float(data):
    """Convert all numeric strings in a nested dictionary to floats."""
    if isinstance(data, dict):
        return {k: convert_to_float(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_float(v) for v in data]
    elif isinstance(data, str):
        try:
            return float(data)
        except ValueError:
            logging.warning(f"Value '{data}' could not be converted to float.")
            return data
    elif isinstance(data, np.float32):
        return float(data)
    return data
