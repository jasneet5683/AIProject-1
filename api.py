import json
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# NEW IMPORTS FOR GOOGLE SHEETS
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# LANGCHAIN / GEMINI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

#for message update
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GET GOOGLE API KEY (For Gemini)
api_key = os.getenv("GOOGLE_API_KEY")

# Global variables
excel_text_context = ""
document_loaded = False
SHEET_NAME = "Task_Manager" # <--- MAKE SURE YOUR GOOGLE SHEET HAS THIS NAME

# --- DATA MODELS ---

class PromptRequest(BaseModel):
    prompt: str

class AddTaskRequest(BaseModel):
    task_name: str
    start_date: str
    end_date: str
    status: str
    assigned_to: str

class UpdateTaskRequest(BaseModel):
    task_name: str
    field_to_update: str
    new_value: str

# --- HELPER: CONNECT TO GOOGLE SHEETS ---
def get_google_sheet():
    """Authenticates and returns the Sheet object."""
    try:
        # Load JSON from Render Environment Variable
        json_creds = os.getenv("GOOGLE_CREDS")
        if not json_creds:
            print("❌ Error: GOOGLE_CREDS not found in environment variables.")
            return None

        creds_dict = json.loads(json_creds)
        
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Open the specific sheet
        sheet = client.open(SHEET_NAME).sheet1
        return sheet
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None

# --- CORE FUNCTIONS ---

def load_data_global():
    """
    Downloads data from Google Sheets instead of local Excel.
    Refreshes the AI's memory.
    """
    global excel_text_context, document_loaded
    
    sheet = get_google_sheet()
    if not sheet:
        document_loaded = False
        return

    try:
        print(f"✅ Fetching data from Google Sheet: {SHEET_NAME}...")
        
        # Get all records as a list of dictionaries
        data = sheet.get_all_records()
        
        if not data:
            print("⚠️ Sheet is empty.")
            excel_text_context = ""
            document_loaded = True
            return

        # Convert to Pandas DataFrame
        df = pd.DataFrame(data)
        
        # Clean Empty Rows
        df.dropna(how='all', inplace=True)
        df = df.fillna("N/A")
        
        # Normalize Dates (Convert to string)
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                except:
                    df[col] = df[col].astype(str)

        # Convert to CSV String for AI Context
        excel_text_context = df.to_csv(index=False)
        
        document_loaded = True
        print(f"✅ Data Loaded! ({len(df)} rows).")
        
    except Exception as e:
        print(f"❌ Error processing data: {str(e)}")
        document_loaded = False

@app.on_event("startup")
async def startup_event():
    load_data_global()

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "Backend is running with Google Sheets integration! ☁️"
    }

@app.get("/api/status")
def get_status():
    return {"document_loaded": document_loaded}

# 1. ADD TASK ENDPOINT (Google Sheets)
@app.post("/api/add-task")
def add_task(request: AddTaskRequest):
    sheet = get_google_sheet()
    if not sheet:
        return {"message": "Could not connect to Google Sheets", "status": "error"}

    try:
        # Prepare the row. ORDER MATTERS! 
        # Ensure your Google Sheet columns are: 
        # [Task Name, Start Date, End Date, Status, Assigned To]
        row_data = [
            request.task_name,
            request.start_date,
            request.end_date,
            request.status,
            request.assigned_to
        ]

        # Append to the bottom of the sheet
        sheet.append_row(row_data)

        # Refresh AI Memory immediately so it knows about the new task
        load_data_global()

        return {"message": "Task saved to Google Sheets!", "status": "success"}

    except Exception as e:
        return {"message": f"Error adding task: {str(e)}", "status": "error"}

# 2. UPDATE TASK ENDPOINT (Google Sheets)
@app.post("/api/update-task")
def update_task(request: UpdateTaskRequest):
    # This is a bit trickier with Sheets.
    # Strategy: Download all -> Update in Pandas -> Clear Sheet -> Upload New Data
    # (This is safest for avoiding row index errors)
    
    sheet = get_google_sheet()
    if not sheet:
        return {"message": "Could not connect to Google Sheets", "status": "error"}

    try:
        # 1. Get Data
        data = sheet.get_all_records()
        df = pd.DataFrame(data)

        # 2. Find the task
        mask = df["Task Name"] == request.task_name
        if not mask.any():
            return {"message": f"Task '{request.task_name}' not found.", "status": "error"}

        # 3. Handle Column Matching
        col_map = {c.strip().lower(): c for c in df.columns}
        target_col = col_map.get(request.field_to_update.strip().lower())

        if not target_col:
            return {"message": f"Column '{request.field_to_update}' not found.", "status": "error"}

        # 4. Update the value in DataFrame
        df.loc[mask, target_col] = request.new_value

        # 5. Save back to Google Sheets
        # clear() removes all data, update([data]) puts new data back
        sheet.clear()
        # Put headers back
        sheet.update([df.columns.values.tolist()] + df.values.tolist())

        # 6. Refresh AI Memory
        load_data_global()

        return {"message": f"Updated '{target_col}' in Google Sheets", "status": "success"}

    except Exception as e:
        return {"message": f"Error updating task: {str(e)}", "status": "error"}

# --- SMART CHAT ENDPOINT WITH TOOLS ---

@tool
def update_sheet_tool(task_name: str, field: str, value: str):
    """
    Updates a task in the Google Sheet.
    Args:
        task_name: The exact name of the task.
        field: The column to update (Status, Assigned To, Start Date, End Date).
        value: The new value to set.
    """
    # Reuse the logic from our update_task endpoint!
    # We create a fake request object to reuse the logic
    req = UpdateTaskRequest(task_name=task_name, field_to_update=field, new_value=value)
    result = update_task(req) # Call the existing python function
    return result["message"]

@app.post("/api/chat")
def chat(request: PromptRequest):
    global excel_text_context
    
    try:
        if not document_loaded:
            load_data_global()

        # 1. Define the Tools the AI can use
        tools = [update_sheet_tool]
        
        # 2. Bind tools to the LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(tools)

        # 3. Create the Conversation Context
        system_msg = f"""
        You are a Project Manager AI. 
        You have access to a tool called 'update_sheet_tool' to modify the Google Sheet.
        
        CURRENT DATA:
        {excel_text_context}
        
        INSTRUCTIONS:
        - If the user asks to UPDATE, CHANGE, or MARK a task, USE THE TOOL.
        - If the user asks a question, just answer from the data.
        - Don't make up confirmation messages unless the tool runs successfully.
        """

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=request.prompt)
        ]

        # 4. Invoke the AI
        ai_response = llm_with_tools.invoke(messages)

        # 5. Check if the AI wants to use a Tool
        if ai_response.tool_calls:
            print("🤖 AI wants to use a tool:", ai_response.tool_calls)
            
            for tool_call in ai_response.tool_calls:
                # Extract arguments provided by AI
                args = tool_call["args"]
                
                # Execute the actual Python function
                tool_result = update_sheet_tool.invoke(args)
                
                # Update Memory
                load_data_global()
                
                return {
                    "response": f"✅ Action Taken: {tool_result}",
                    "type": "text",
                    "status": "success"
                }

        # 6. If no tool needed, just return the text response
        return {
            "response": ai_response.content,
            "type": "text",
            "status": "success"
        }

    except Exception as e:
        return {"response": f"Error: {str(e)}", "status": "error"}
        
# 3. CHAT ENDPOINT (Unchanged logic, uses new global context)
# @app.post("/api/chat")
# def chat(request: PromptRequest):
#    global excel_text_context
    
 #   try:
 #       if not document_loaded:
  #          # Try reloading one more time if missing
  #          load_data_global()
  #          if not document_loaded:
  #              return {"response": "System is offline or Google Sheet not connected.", "status": "error"}

   #     # SYSTEM INSTRUCTION
    #    system_instruction = f"""
    #    You are a Data Analyst AI. 
    #    Below is the Project Data from Google Sheets:
        
     #   --- DATA START ---
     #   {excel_text_context}
     #   --- DATA END ---

     #   INSTRUCTIONS:
     #   1. Analyze the data above to answer the user's question.
     #   2. Be precise with numbers, dates, and names.
        
      #  OUTPUT FORMATS (Strict JSON):
     #   A) FOR TABLES/LISTS: {{ "is_table": true, "title": "...", "columns": [...], "rows": [...], "summary": "..." }}
     #   B) FOR CHARTS: {{ "is_chart": true, "chart_type": "bar", "title": "...", "data": {{ "labels": [...], "values": [...] }}, "summary": "..." }}
     #   C) FOR TEXT: Return plain text.
     #   """

     #   llm = ChatGoogleGenerativeAI(
     #       model="gemini-2.5-flash",  
     #       google_api_key=api_key,
     #       temperature=0
     #   )
        
      #  full_prompt = f"{system_instruction}\n\nUser Question: {request.prompt}"
        
      #  response = llm.invoke(full_prompt)
      #  content = response.content.strip()
        
      #  # Clean Markdown
      #  clean_content = content.replace("```json", "").replace("```", "").strip()

      #  try:
      #      data_obj = json.loads(clean_content)
      #      if data_obj.get("is_chart") is True:
      #          return {"response": data_obj["summary"], "chart_data": data_obj, "type": "chart", "status": "success"}
      #      if data_obj.get("is_table") is True:
      #           return {"response": data_obj["summary"], "table_data": data_obj, "type": "table", "status": "success"}
      #  except:
      #      pass

      #  return {"response": clean_content, "type": "text", "status": "success"}

   # except Exception as e:
   #     return {"response": f"Internal Error: {str(e)}", "status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    


