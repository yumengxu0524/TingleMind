import os
from openai import OpenAI
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

openai_client = OpenAI(api_key=openai_api_key)
import httpx
from motor.motor_asyncio import AsyncIOMotorClient

from fastapi import FastAPI, WebSocket, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime
from dateutil import parser
from typing import Optional, List, Tuple
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocketDisconnect
import json

import logging
import traceback
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('pymongo').setLevel(logging.WARNING)

# Import from your psychological analysis module
from Tingle_Brain_A_Agent_One import (nlp_model, expanded_dict, calculate_daily_scores, handle_follow_up_question,prepare_score_changes_for_mongo,
                                      track_score_changes, accumulate_scores, get_top_traits, determine_tone, prepare_context_for_agent_1, 
                                      prepare_context_for_agent_2,get_analysis_from_chatgpt)
# MongoDB connection
client = AsyncIOMotorClient("mongodb://localhost:27017")
db = client['mydatabase']
collection = db['entries']

# FastAPI application
app = FastAPI()

# Serve static files (for frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Request model
class Entry(BaseModel):
    user_id: str
    title: str
    content: str
    time: Optional[str] = None

# Response models
class AnalysisDetails(BaseModel):
    summary: str
    top_traits: List[Tuple[str, float]]  # Now accepts tuples of (trait_name, trait_score)
    accumulated_scores: dict

class ResponseModel(BaseModel):
    message: str
    analysis: Optional[str] = None
    details: Optional[AnalysisDetails] = None

class DiaryEntry(BaseModel):
    title: str
    content: str
    time: str    

class UserDataResponse(BaseModel):
    user_id: str
    entries: List[DiaryEntry]   

# Model for follow-up question
class FollowUpRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    # Serve the index.html file
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Helper function to save context agent 1 data to MongoDB
async def save_context_agent_1_data(user_id, analysis, diary_entries, daily_scores, accumulated_scores, top_traits, tone, score_changes, entry_time):
    # Prepare context data for context agent 1
    context_data = {
        "user_id": user_id,
        "analysis": analysis,
        "diary_entries": diary_entries,
        "daily_scores": daily_scores,
        "accumulated_scores": accumulated_scores,
        "top_traits": top_traits,
        "score_changes": score_changes,  # Include the tracked score changes
        "tone": tone,
        "time": entry_time
    }

    # Save context agent 1 data in MongoDB (asynchronously)
    await db['context_agent_data'].update_one(
        {"user_id": user_id},
        {"$set": context_data},
        upsert=True
    )

# Submit entry function (integrating the save_context_agent_1_data)
@app.post("/submit_entry/")
async def submit_entry(entry: Entry):
    try:
        logging.debug("Inserting entry into MongoDB")
        
        # Parse time or use current date if not provided
        entry_time = parser.parse(entry.time).strftime("%Y-%m-%d") if entry.time else datetime.now().strftime("%Y-%m-%d")

        # Insert entry into MongoDB (asynchronously)
        result = await collection.insert_one({
            "user_id": entry.user_id,
            "title": entry.title,
            "content": entry.content,
            "time": entry_time
        })
        
        if result:
            logging.debug(f"Entry inserted with ID: {str(result.inserted_id)} for user: {entry.user_id}")

            # Fetch diary entries from MongoDB (asynchronously)
            diary_entries = await fetch_diary_entries(entry.user_id)

            # Perform psychological analysis
            daily_scores = calculate_daily_scores(diary_entries, nlp_model, expanded_dict)
            tracked_score_changes = track_score_changes(daily_scores)  # Track changes in emotional scores
            score_changes_for_mongo = prepare_score_changes_for_mongo(tracked_score_changes)  # Prepare score changes

            accumulated_scores = accumulate_scores(daily_scores)
            top_traits = get_top_traits(accumulated_scores)
            tone = determine_tone(daily_scores[max(diary_entries.keys())])

            # Prepare the context for context_agent_1 (initial psychological analysis)
            context_agent_1 = prepare_context_for_agent_1(diary_entries, daily_scores, accumulated_scores, top_traits)

            # Fetch analysis from OpenAI (asynchronously)
            analysis = await get_analysis_from_chatgpt(context_agent_1)
            
            # Convert date keys to string for MongoDB compatibility
            diary_entries_str = {date.strftime("%Y-%m-%d"): content for date, content in diary_entries.items()}
            daily_scores_str = prepare_daily_scores_for_mongo(daily_scores)

            # Use the helper function to save all context agent 1 data
            await save_context_agent_1_data(
                entry.user_id,
                analysis,
                diary_entries_str,
                daily_scores_str,
                accumulated_scores,
                top_traits,
                tone,
                score_changes_for_mongo,  # Save score changes in MongoDB
                entry_time
            )

            # Build the response
            details = AnalysisDetails(
                summary="Summary of the analysis",
                top_traits=top_traits,
                accumulated_scores=accumulated_scores
            )

            return ResponseModel(
                message="Entry submitted and analyzed successfully",
                analysis=analysis,
                details=details
            )

        else:
            logging.error("Entry submission failed")
            raise HTTPException(status_code=500, detail="Entry submission failed")
        
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for follow-up questions
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    await websocket.send_text(f"Connected to the Tingle chat. User ID: {user_id}. You can now ask follow-up questions. Type 'exit' to close the chat.")

    try:
        while True:
            data = await websocket.receive_text()

            if data.lower() == "exit":
                await websocket.send_text("Goodbye!")
                await websocket.close()
                break

            # Handle the follow-up question using the provided user_id
            logging.debug(f"Message received from WebSocket: {data}")
            follow_up_response = await handle_follow_up_question(websocket, user_id, data)

            # Send the follow-up answer
            await websocket.send_text(f"ChatGPT Response: {follow_up_response}")
    
    except WebSocketDisconnect:
        logging.info(f"User {user_id} disconnected")
    except Exception as e:
        await websocket.send_text(f"Error occurred: {str(e)}")
        await websocket.close()


@app.get("/data/{user_id}", response_model=UserDataResponse)
async def get_data(user_id: str):
    try:
        logging.debug(f"Fetching data for user_id: {user_id}")
        
        # Fetch the data from MongoDB, excluding the MongoDB internal _id field
        data = await collection.find({"user_id": user_id}, {"_id": 0}).to_list(length=250) # This will retrieve up to 250 documents from the MongoDB collection and store them as a list. Adjust the length parameter as needed.

        
        # Check if no data was found
        if not data:
            logging.debug(f"No data found for user_id: {user_id}")
            raise HTTPException(status_code=404, detail="No data found for user")
        
        logging.debug(f"Data found for user_id: {user_id}")
        
        # Build response format
        entries = [DiaryEntry(title=entry["title"], content=entry["content"], time=entry["time"]) for entry in data]
        
        return UserDataResponse(user_id=user_id, entries=entries)
    
    except Exception as e:
        logging.error(f"Exception occurred while fetching data for user_id: {user_id} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    

async def fetch_diary_entries(user_id: str): 
    entries = await collection.find({"user_id": user_id}).to_list(length=200)
    diary_entries = {}
    
    for entry in entries:
        try:
            # Check if both 'time' and 'content' fields are present
            if 'time' not in entry or 'content' not in entry:
                logging.warning(f"Missing 'time' or 'content' in entry: {entry}")
                continue

            # Convert the 'time' field from string to datetime.date object
            diary_date = datetime.strptime(entry['time'], "%Y-%m-%d").date()

            # Append the content to the respective date in diary_entries
            if diary_date not in diary_entries:
                diary_entries[diary_date] = []
            diary_entries[diary_date].append(entry['content'])

        except ValueError as ve:
            logging.error(f"ValueError processing entry {entry}: {ve}")
            continue
        except Exception as e:
            logging.error(f"Unexpected error processing entry {entry}: {e}")
            continue
    
    # Log final results for debugging
    logging.debug(f"Diary entries fetched for user {user_id}: {diary_entries}")

    return diary_entries if diary_entries else {}



def prepare_daily_scores_for_mongo(daily_scores):
    # Convert all datetime.date keys to strings
    daily_scores_str = {date.strftime("%Y-%m-%d"): scores for date, scores in daily_scores.items()}
    return daily_scores_str

# Function to get latest entry and tone (used for context agent 2)
async def get_latest_entry_and_tone(user_id: str, daily_scores: dict):
    diary_entries = await fetch_diary_entries(user_id)
    if not diary_entries:
        raise HTTPException(status_code=400, detail="No diary entries found for user.")
        
    latest_entry_date = max(diary_entries.keys())
    tone = determine_tone(daily_scores[latest_entry_date.strftime("%Y-%m-%d")])
    return latest_entry_date, tone

# create an endpoint to retrieve the stored data for monitoring purposes:
@app.get("/context_agent_data/{user_id}")
async def get_context_agent_data(user_id: str):
    try:
        context_data = await db['context_agent_data'].find_one({"user_id": user_id}, {"_id": 0})
        if not context_data:
            return {"message": "No data found for user"}
        return context_data
    except Exception as e:
        logging.error(f"Exception occurred while fetching context data for user_id: {user_id} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def handle_follow_up_question(websocket: WebSocket, user_id: str, user_question: str):
    try:
        # Retrieve stored context data from MongoDB
        context_data = await db['context_agent_data'].find_one({"user_id": user_id}, {"_id": 0})
        if not context_data:
            await websocket.send_text("No prior analysis found. Please submit a diary entry first.")
            return

        # Extract context data for context agent 2
        analysis = context_data.get('analysis')
        diary_entries = context_data.get('diary_entries')
        daily_scores = context_data.get('daily_scores', {})

        accumulated_scores = context_data.get('accumulated_scores')
        top_traits = context_data.get('top_traits')

        # Log the extracted data to ensure it was retrieved correctly
        logging.debug(f"Diary Entries: {diary_entries}")
        logging.debug(f"Daily Scores being sent to context agent 2: {daily_scores}")
        logging.debug(f"Accumulated Scores: {accumulated_scores}")
        logging.debug(f"Top Traits: {top_traits}")
        
        # Fetch the latest diary entry and tone (for context_agent_2 only)
        latest_entry_date, tone = await get_latest_entry_and_tone(user_id, daily_scores)

        # Prepare context agent 2 with updated context (including tone)
        context_agent_2 = prepare_context_for_agent_2(diary_entries, daily_scores, accumulated_scores, top_traits, tone)

        # Log the context being sent to agent 2 for debugging
        logging.debug(f"Context being sent to agent 2: {context_agent_2}")

        # Prepare system prompt for context agent 2
        system_prompt_2 = (
            "You are an assistant helping answer follow-up questions on a psychological analysis of diary entries. "
            "You are a psychologist analyzing diary entries using a framework that includes tracking emotional scores, "
            "cumulative trait scores, and identifying key psychological traits over time. The framework analyzes traits "
            "such as cautiousness, happiness, sadness, resilience, social connection, and self-esteem, and calculates changes over time. "
            "Please analyze trends in emotional states, identify recurring themes, and provide insights on the writer’s psychological traits and overall mental well-being. "
            "Focus on how the individual’s emotional state fluctuates, suggesting areas for personal growth based on the most prominent traits. "
            "Use the numerical scores provided to interpret the intensity of each trait, noting any significant increases or decreases."
        )

        # Build messages for ChatGPT with context agent 2 and follow-up question
        messages = [
            {"role": "system", "content": system_prompt_2},
            {"role": "assistant", "content": analysis},  # Include the prior analysis
            {"role": "assistant", "content": context_agent_2},  # Include the context with daily_scores
            {"role": "user", "content": user_question}  # The current follow-up question
        ]

        # Log the messages being sent to the OpenAI API for debugging
        logging.debug(f"Messages being sent to OpenAI API: {json.dumps(messages, indent=3)}")

        # Send the follow-up question to ChatGPT asynchronously using httpx
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_api_key}"},
                json={
                    "model": "gpt-4",
                    "messages": messages
                }
            )
        
        # Check if the response is successful
        if response.status_code != 200:
            error_message = f"OpenAI API Error: {response.status_code} - {response.text}"
            logging.error(error_message)
            await websocket.send_text(f"Error: Unable to retrieve follow-up response. {response.text}")
            return
        
        # Parse the response JSON
        try:
            result = response.json()
            logging.debug(f"Response received: {json.dumps(result, indent=2)}")

        except json.JSONDecodeError as json_error:
            logging.error(f"Failed to decode JSON from OpenAI API response: {json_error}")
            logging.error(f"Raw response text: {response.text}")
            await websocket.send_text("Error: Failed to parse response from the server. Please try again later.")
            return

        # Ensure 'choices' is in the response
        if "choices" not in result or not result['choices']:
            logging.error("Unexpected response format: 'choices' key missing or empty.")
            await websocket.send_text("Error: No valid response from ChatGPT.")
            return

        # Extract the response from ChatGPT
        follow_up_answer = result['choices'][0]['message']['content'].strip() if result.get('choices') else None

        # Ensure the follow-up answer is not None before sending it
        if follow_up_answer:
            logging.debug(f"Follow-up answer received from ChatGPT: {follow_up_answer}")
            await websocket.send_text(f"ChatGPT Response: {follow_up_answer}")
        else:
            logging.error("No response from ChatGPT.")
            await websocket.send_text("ChatGPT was unable to provide a response.")

        # Store the follow-up answer in MongoDB asynchronously for future follow-up references
        if follow_up_answer:
            await db['context_agent_follow_ups'].update_one(
                {"user_id": user_id},
                {"$set": {"previous_follow_up_answer": follow_up_answer}},
                upsert=True
            )

    except Exception as e:
        logging.error(f"Error occurred in handle_follow_up_question: {str(e)}")
        
        # Ensure response is defined before checking its content
        if 'response' in locals() and response is not None:
            logging.error(f"Full error response: {response.text}")
        else:
            logging.error("No response received from OpenAI API")
        
        # Log the full traceback
        logging.error(traceback.format_exc())  # Log full traceback for better diagnostics
        
        # Send a generic error message to the WebSocket
        await websocket.send_text(f"Error occurred: {str(e)}. Please try again later.")



@app.get("/context_agent_2_data/{user_id}")
async def get_context_agent_2_data(user_id: str):
    try:
        context_data = await db['context_agent_data'].find_one({"user_id": user_id}, {"_id": 0, "context_agent_2": 1})
        if not context_data:
            return {"message": "No data found for user"}
        return context_data
    except Exception as e:
        logging.error(f"Error occurred while fetching context agent 2 data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
