import json
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

# OPENAI IMPORTS
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

# API KEY
openai_key = os.getenv("OPENAI_API_KEY")

# Global variables
excel_text_context = ""
document_loaded = False
SHEET_NAME = "Task_Manager"

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
    return {"status": "active", "message": "Backend is running with OpenAI + Tools + Charts! 📊"}

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

# 2. UPDATE TASK (Robust Version)
@app.post("/api/update-task")
def update_task_endpoint(request: UpdateTaskRequest):
    return internal_update_task(request.task_name, request.field_to_update, request.new_value)

def internal_update_task(task_name, field, value):
    sheet = get_google_sheet()
    if not sheet:
        return {"message": "Connection Error", "status": "error"}

    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)

        # Robust Column Mapping
        col_map = {c.strip().lower().replace("_", " "): c for c in df.columns}
        
        task_col_actual = col_map.get("task name") or col_map.get("taskname") or col_map.get("task")
        if not task_col_actual:
            return {"message": "Could not find 'Task Name' column", "status": "error"}

        target_col_clean = field.strip().lower().replace("_", " ")
        target_col_actual = col_map.get(target_col_clean)
        if not target_col_actual:
            return {"message": f"Column '{field}' not found.", "status": "error"}

        mask = df[task_col_actual].astype(str).str.strip().str.lower() == task_name.strip().lower()
        if not mask.any():
            return {"message": f"Task '{task_name}' not found.", "status": "error"}

        df.loc[mask, target_col_actual] = value
        
        sheet.clear()
        sheet.update([df.columns.values.tolist()] + df.values.tolist())
        load_data_global()
        return {"message": f"✅ Updated '{task_name}': Set '{target_col_actual}' to '{value}'", "status": "success"}

    except Exception as e:
        return {"message": f"Error updating: {str(e)}", "status": "error"}


# --- 3. SMART CHAT AGENT (TOOLS + CHARTS) ---

@tool
def update_sheet_tool(task_name: str, field: str, value: str):
    """
    Updates a task in the Google Sheet. Use this ONLY when the user explicitly asks to modify/update data.
    """
    print(f"🛠 Tool Triggered: Updating {task_name}...")
    result = internal_update_task(task_name, field, value)
    return result["message"]

@app.post("/api/chat")
def chat(request: PromptRequest):
    global excel_text_context
    
    try:
        if not document_loaded:
            load_data_global()

        tools = [update_sheet_tool]
        
        llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=openai_key,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(tools)

        # --- SYSTEM PROMPT: INCLUDES CHARTS & TABLES ---
        system_msg = f"""
        You are a smart Project Manager Assistant.
        
        CURRENT DATA:
        {excel_text_context}
        
        INSTRUCTIONS:
        1. **ACTION**: If the user asks to UPDATE/CHANGE data, call the 'update_sheet_tool'.
        
        2. **VISUALIZATION**: If the user asks for a TABLE, LIST, or CHART/GRAPH, you MUST return a strict JSON object (Markdown formatted).
           
           - FORMAT FOR TABLE:
             ```json
             {{
               "is_table": true,
               "title": "Task List",
               "columns": ["Task Name", "Status", "Assigned To"],
               "rows": [["Design", "Done", "John"], ["Dev", "Pending", "Jane"]],
               "summary": "Here is the list of tasks."
             }}
             ```
             
           - FORMAT FOR CHART:
             ```json
             {{
               "is_chart": true,
               "chart_type": "bar", 
               "title": "Tasks by Status",
               "data": {{ "labels": ["Done", "Pending"], "values": [5, 2] }},
               "summary": "Here is the breakdown by status."
             }}
             ```
             (chart_type can be: bar, pie, line, doughnut)

        3. **TEXT**: Otherwise, answer normally in plain text.
        """

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=request.prompt)
        ]

        # Invoke LLM
        print("🤖 AI Thinking...")
        ai_response = llm_with_tools.invoke(messages)

        # --- CASE A: TOOL CALL (UPDATE) ---
        if ai_response.tool_calls:
            print("🔧 AI using Tool:", ai_response.tool_calls)
            for tool_call in ai_response.tool_calls:
                selected_tool = {"update_sheet_tool": update_sheet_tool}[tool_call["name"].lower()]
                tool_output = selected_tool.invoke(tool_call["args"])
                return {
                    "response": f"✅ {tool_output}",
                    "type": "text",
                    "status": "success"
                }

        # --- CASE B: CHECK FOR JSON (CHARTS/TABLES) ---
        content = ai_response.content.strip()
        
        if "```json" in content:
            # Extract JSON from Markdown
            try:
                clean_json = content.split("```json")[1].split("```")[0].strip()
                data_obj = json.loads(clean_json)
                
                if data_obj.get("is_chart"):
                    return {"response": data_obj["summary"], "chart_data": data_obj, "type": "chart", "status": "success"}
                
                if data_obj.get("is_table"):
                    return {"response": data_obj["summary"], "table_data": data_obj, "type": "table", "status": "success"}
            except Exception as e:
                print(f"JSON Parse Error: {e}")
                # Fallback to returning raw text if parsing fails
                pass

        # --- CASE C: REGULAR TEXT ---
        return {
            "response": content,
            "type": "text",
            "status": "success"
        }

    except Exception as e:
        print(f"❌ Chat Error: {e}")
        return {"response": f"Error: {str(e)}", "status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```*
