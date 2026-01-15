import json
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

# ✅ CHANGED: OpenAI Imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

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

# ✅ CHANGED: Get OpenAI API Key
openai_key = os.getenv("OPENAI_API_KEY")

# Global variables
excel_text_context = ""
document_loaded = False
SHEET_NAME = "Task_Manager"  # Ensure this matches your Sheet Name

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
    try:
        json_creds = os.getenv("GOOGLE_CREDS")
        if not json_creds:
            print("❌ Error: GOOGLE_CREDS not found.")
            return None
        creds_dict = json.loads(json_creds)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open(SHEET_NAME).sheet1
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None

# --- CORE FUNCTIONS ---

def load_data_global():
    global excel_text_context, document_loaded
    sheet = get_google_sheet()
    if not sheet:
        document_loaded = False
        return

    try:
        data = sheet.get_all_records()
        if not data:
            excel_text_context = ""
            document_loaded = True
            return

        df = pd.DataFrame(data)
        df.fillna("N/A", inplace=True)
        
        # Normalize dates
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                except:
                    df[col] = df[col].astype(str)

        excel_text_context = df.to_csv(index=False)
        document_loaded = True
        print("✅ Data Loaded/Refreshed.")
        
    except Exception as e:
        print(f"❌ Error processing data: {str(e)}")
        document_loaded = False

@app.on_event("startup")
async def startup_event():
    load_data_global()

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"status": "active", "message": "Backend is running with OpenAI! 🧠"}

@app.get("/api/status")
def get_status():
    return {"document_loaded": document_loaded}

# 1. ADD TASK
@app.post("/api/add-task")
def add_task(request: AddTaskRequest):
    sheet = get_google_sheet()
    if not sheet:
        return {"message": "Connection Error", "status": "error"}
    try:
        row = [request.task_name, request.start_date, request.end_date, request.status, request.assigned_to]
        sheet.append_row(row)
        load_data_global()
        return {"message": "Task saved successfully!", "status": "success"}
    except Exception as e:
        return {"message": f"Error: {str(e)}", "status": "error"}

# 2. UPDATE TASK (Internal Logic)
@app.post("/api/update-task")
def update_task_endpoint(request: UpdateTaskRequest):
    result = internal_update_task(request.task_name, request.field_to_update, request.new_value)
    return result

def internal_update_task(task_name, field, value):
    """Updates the sheet and returns a status dict."""
    sheet = get_google_sheet()
    if not sheet:
        return {"message": "Connection Error", "status": "error"}

    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)

        # Find task (Case insensitive search)
        mask = df["Task Name"].astype(str).str.lower() == task_name.lower()
        if not mask.any():
            return {"message": f"Task '{task_name}' not found.", "status": "error"}

        # Find column
        col_map = {c.strip().lower(): c for c in df.columns}
        target_col = col_map.get(field.strip().lower())
        if not target_col:
            return {"message": f"Column '{field}' not found.", "status": "error"}

        # Update
        df.loc[mask, target_col] = value
        
        # Save to Sheet (Clear & Rewrite is safest)
        sheet.clear()
        sheet.update([df.columns.values.tolist()] + df.values.tolist())
        
        load_data_global() # Refresh Memory
        return {"message": f"Updated '{task_name}' - set {target_col} to '{value}'", "status": "success"}

    except Exception as e:
        return {"message": f"Error updating: {str(e)}", "status": "error"}


# --- 3. SMART CHAT AGENT (OPENAI VERSION) ---

@tool
def update_sheet_tool(task_name: str, field: str, value: str):
    """
    Updates a task in the Google Sheet.
    Args:
        task_name: The exact name of the task (e.g., 'Design Homepage').
        field: The column name to update (Status, Assigned To, Start Date, End Date).
        value: The new value to set (e.g., 'Completed', 'John Doe').
    """
    print(f"🛠 Tool Triggered: Updating {task_name} | {field} -> {value}")
    result = internal_update_task(task_name, field, value)
    return result["message"]

@app.post("/api/chat")
def chat(request: PromptRequest):
    global excel_text_context
    
    try:
        # Reload if memory is empty
        if not document_loaded:
            load_data_global()

        # 1. Define Tools
        tools = [update_sheet_tool]
        
        # 2. Initialize OpenAI LLM
        # using 'gpt-4o' or 'gpt-3.5-turbo' (both support tools well)
        llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=openai_key,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(tools)

        # 3. System Prompt
        system_msg = f"""
        You are a Project Manager Assistant.
        
        CURRENT DATA:
        {excel_text_context}
        
        INSTRUCTIONS:
        - If the user asks to UPDATE, CHANGE, or MODIFY a task, YOU MUST USE the 'update_sheet_tool'.
        - If the user asks a question, answer from the data.
        - Do NOT hallucinate. Only confirm update if the tool is called.
        """

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=request.prompt)
        ]

        # 4. Invoke LLM
        print("🤖 AI Thinking (OpenAI)...")
        ai_response = llm_with_tools.invoke(messages)

        # 5. Handle Tool Call
        if ai_response.tool_calls:
            print("🔧 AI decided to use a tool:", ai_response.tool_calls)
            
            for tool_call in ai_response.tool_calls:
                # Map the tool name to the function
                selected_tool = {"update_sheet_tool": update_sheet_tool}[tool_call["name"].lower()]
                
                # Run the tool
                tool_output = selected_tool.invoke(tool_call["args"])
                
                # Return result immediately
                return {
                    "response": f"✅ {tool_output}",
                    "type": "text",
                    "status": "success"
                }

        # 6. Regular Text Response
        return {
            "response": ai_response.content,
            "type": "text",
            "status": "success"
        }

    except Exception as e:
        print(f"❌ Chat Error: {e}")
        return {"response": f"Error: {str(e)}", "status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
