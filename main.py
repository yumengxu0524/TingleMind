import os
from openai import OpenAI
openai_client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="enter_your_OpenAI_api_key"
)
from fastapi import FastAPI, WebSocket, HTTPException, Request
from pydantic import BaseModel
from pymongo import MongoClient
from datetime import datetime
from dateutil import parser
from typing import Optional, List
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Import from your psychological analysis module
from Tingle_Brain_A_Agent_One import (nlp_model, expanded_dict, calculate_daily_scores, follow_up_questions,
                                      track_score_changes, accumulate_scores, get_top_traits, determine_tone, prepare_context_for_agent_1, 
                                      prepare_context_for_agent_2,get_analysis_from_chatgpt)
# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client['mydatabase']
collection = db['entries']

# FastAPI application
app = FastAPI()

# Serve static files (for frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")


class Entry(BaseModel):
    user_id: str                                        
    title: str
    content: str
    time: Optional[str] = None

@app.get("/")
def root():
    # Serve the index.html file
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/submit_entry/")
def submit_entry(entry: Entry):
    try:
        logging.debug("Inserting entry into MongoDB")
        # Parse time or use current date if not provided
        entry_time = parser.parse(entry.time).strftime("%Y-%m-%d") if entry.time else datetime.now().strftime("%Y-%m-%d")

        # Insert entry into MongoDB
        logging.debug(f"Inserting entry for user_id: {entry.user_id}, title: {entry.title}")
        result = collection.insert_one({
            "user_id": entry.user_id,
            "title": entry.title,
            "content": entry.content,
            "time": entry_time
        })
        logging.debug("Inserted into MongoDB with result: ", result)

        if result:
            logging.debug(f"Entry inserted with ID: {str(result.inserted_id)} for user: {entry.user_id}")

            # Fetch diary entries from MongoDB
            logging.debug("Fetching diary entries for analysis")
            diary_entries = fetch_diary_entries(entry.user_id)

            logging.debug("Performing psychological analysis")            
            # Calculate daily scores
            daily_scores = calculate_daily_scores(diary_entries, nlp_model, expanded_dict)

            # Get the accumulated scores and top traits
            accumulated_scores = accumulate_scores(daily_scores)
            top_traits = get_top_traits(accumulated_scores)
            latest_entry_date = max(diary_entries.keys())
            tone = determine_tone(daily_scores[latest_entry_date])

            # Prepare the context for context_agent_1 (initial psychological analysis)
            context_agent_1 = prepare_context_for_agent_1(diary_entries, daily_scores, accumulated_scores, top_traits)
            logging.debug("Calling get_analysis_from_chatgpt with context_agent_1")
            analysis = get_analysis_from_chatgpt(context_agent_1)
            logging.debug(f"Analysis response received: {analysis}")

            # Store user_id and analysis context for later use in follow-up questions
            app.state.user_id = entry.user_id
            app.state.context_agent_1_analysis = analysis
            app.state.context_agent_1_diary_entries = diary_entries
            app.state.context_agent_1_daily_scores = daily_scores
            app.state.context_agent_1_accumulated_scores = accumulated_scores
            app.state.context_agent_1_top_traits = top_traits

            # Return the analysis and success message
            return {"message": "Entry submitted and analyzed successfully", "analysis": analysis}
        else:
            logging.error("Entry submission failed")
            raise HTTPException(status_code=500, detail="Entry submission failed")
    
    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        # Handle exceptions gracefully with a detailed message
        raise HTTPException(status_code=500, detail=str(e))

# Model for follow-up question
class FollowUpRequest(BaseModel):
    question: str


@app.post("/ask_follow_up/{user_id}")
def ask_follow_up(user_id: str, request: FollowUpRequest):
    try:
        user_question = request.question

        # Retrieve stored context from app.state
        if not hasattr(app.state, 'context_agent_1_analysis') or not hasattr(app.state, 'context_agent_1_diary_entries'):
            raise HTTPException(status_code=400, detail="No prior analysis found. Please submit a diary entry first.")
        
        # Retrieve user_id and analysis context stored during diary submission
        analysis = app.state.context_agent_1_analysis
        diary_entries = app.state.context_agent_1_diary_entries
        daily_scores = app.state.context_agent_1_daily_scores
        accumulated_scores = app.state.context_agent_1_accumulated_scores
        top_traits = app.state.context_agent_1_top_traits

        # Fetch the latest diary entry and tone (for context_agent_2 only)
        latest_entry_date, tone = get_latest_entry_and_tone(user_id, daily_scores)

        # Prepare context agent 2 with updated context (including tone)
        context_agent_2 = prepare_context_for_agent_2(diary_entries, daily_scores, accumulated_scores, top_traits, tone)

        # Send the follow-up question to ChatGPT with full context for context agent 2
        follow_up_response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant helping answer follow-up questions on a psychological analysis of diary entries."},
                {"role": "assistant", "content": analysis},
                {"role": "assistant", "content": context_agent_2},
                {"role": "user", "content": user_question}
            ]
        )

        # Extract the response from ChatGPT
        follow_up_answer = follow_up_response.choices[0].message.content
        
        # Return the follow-up answer as the response
        return {"answer": follow_up_answer}

    except Exception as e:
        logging.error(f"Exception occurred during follow-up: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/data/{user_id}")
def get_data(user_id: str):
    try:
        logging.debug(f"Fetching data for user_id: {user_id}")        
        data = list(collection.find({"user_id": user_id}, {"_id": 0}))  # Exclude MongoDB _id from the result
        if not data:
            logging.debug(f"No data found for user_id: {user_id}")            
            return {"message": "No data found for user"}
        logging.debug(f"Data found for user_id: {user_id}")       
        return data
    except Exception as e:
        logging.error(f"Exception occurred while fetching data for user_id: {user_id} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/data/{user_id}")
def get_data(user_id: str):
    try:
        logging.debug(f"Fetching data for user_id: {user_id}")        
        data = list(collection.find({"user_id": user_id}, {"_id": 0}))  # Exclude MongoDB _id from the result
        if not data:
            logging.debug(f"No data found for user_id: {user_id}")            
            return {"message": "No data found for user"}
        logging.debug(f"Data found for user_id: {user_id}")       
        return data
    except Exception as e:
        logging.error(f"Exception occurred while fetching data for user_id: {user_id} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# WebSocket for real-time chat
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Connected to the Tingle chat. You can now ask follow-up questions.")
    
    while True:
        data = await websocket.receive_text()
        if data.lower() == "exit":
            await websocket.send_text("Goodbye!")
            break

        # Placeholder response for follow-up questions
        response = f"Received: {data}. Analysis will be provided soon."
        await websocket.send_text(response)

# Function to fetch diary entries from MongoDB
def fetch_diary_entries(user_id: str):
    logging.debug(f"Fetching diary entries for user_id: {user_id}")    
    entries = collection.find({"user_id": user_id})
    diary_entries = {}
    for entry in entries:
        diary_entries[datetime.strptime(entry['time'], '%Y-%m-%d').date()] = [entry['content']]
    return diary_entries

# Function to get latest entry and tone (used for context agent 2)
def get_latest_entry_and_tone(user_id: str, daily_scores: dict):
    diary_entries = fetch_diary_entries(user_id)
    latest_entry_date = max(diary_entries.keys())
    tone = determine_tone(daily_scores[latest_entry_date])
    return latest_entry_date, tone



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
