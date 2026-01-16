import json
import os
import requests
# import smtplib
import pandas as pd
# from email.message import EmailMessage
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

# --- CONFIGURATION ---
openai_key = os.getenv("OPENAI_API_KEY")
email_sender = os.getenv("EMAIL_SENDER")
email_password = os.getenv("EMAIL_PASSWORD")
SHEET_NAME = "Task_Manager"  # Make sure this matches your Google Sheet Name exactly

# Global variables to hold data state
excel_text_context = ""
document_loaded = False

# --- DATA MODELS ---
class PromptRequest(BaseModel):
    prompt: str

# --- 1. HELPER: CONNECT TO GOOGLE SHEETS ---
def get_google_sheet():
    try:
        json_creds = os.getenv("GOOGLE_CREDS")
        if not json_creds:
            print("❌ Error: GOOGLE_CREDS not found in environment.")
            return None
        creds_dict = json.loads(json_creds)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open(SHEET_NAME).sheet1
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None

# --- 2. HELPER: LOAD DATA (Fixes 'Sheet is Empty') ---
def load_data_global():
    global excel_text_context, document_loaded
    print("🔄 Loading data from Google Sheets...")
    sheet = get_google_sheet()
    if not sheet:
        document_loaded = False
        return

    try:
        data = sheet.get_all_records()
        if not data:
            print("⚠️ Sheet is empty or couldn't read records.")
            excel_text_context = "No data found."
            document_loaded = True
            return

        df = pd.DataFrame(data)
        df.fillna("N/A", inplace=True)
        
        # Convert dates to string to avoid errors
        for col in df.columns:
            if "date" in col.lower():
                df[col] = df[col].astype(str)

        excel_text_context = df.to_csv(index=False)
        document_loaded = True
        print("✅ Data Successfully Loaded into Memory.")
        
    except Exception as e:
        print(f"❌ Error processing data: {str(e)}")
        document_loaded = False

# --- 3. HELPER: EMAIL SENDER (VIA API - WORKS ON RENDER) ---
def internal_send_email(to_email, subject, body):
    api_key = os.getenv("BREVO_API_KEY")
    sender_email = os.getenv("SENDER_EMAIL")
    sender_name = os.getenv("SENDER_NAME")

    if not api_key:
        return {"message": "❌ Missing BREVO_API_KEY in .env", "status": "error"}

    url = "https://api.brevo.com/v3/smtp/email"
    
    payload = {
        "sender": {"name": sender_name, "email": sender_email},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": f"<p>{body}</p>"
    }
    
    headers = {
        "accept": "application/json",
        "api-key": api_key,
        "content-type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code == 201:
            return {"message": f"✅ Email sent to {to_email} successfully!", "status": "success"}
        else:
            return {"message": f"❌ Failed: {response.text}", "status": "error"}
            
    except Exception as e:
        return {"message": f"❌ Error: {str(e)}", "status": "error"}

# --- 4. HELPER: UPDATE TASK ---
def internal_update_task(task_name, field, value):
    sheet = get_google_sheet()
    if not sheet:
        return {"message": "Connection Error", "status": "error"}

    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)

        # Flexible column matching
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
        load_data_global() # Refresh memory after update
        return {"message": f"✅ Updated '{task_name}': Set '{target_col_actual}' to '{value}'", "status": "success"}

    except Exception as e:
        return {"message": f"Error updating: {str(e)}", "status": "error"}


# --- 5. APP STARTUP EVENT ---
@app.on_event("startup")
async def startup_event():
    # This runs when Render starts the server
    load_data_global()

# --- 6. API ENDPOINTS ---

# Fixes 404 Error
@app.get("/")
def read_root():
    return {"status": "active", "message": "Backend is running. Data loaded: " + str(document_loaded)}

@app.get("/api/status")
def get_status():
    return {"document_loaded": document_loaded, "data_preview": excel_text_context[:100]}

# --- 7. LANGCHAIN TOOLS ---

@tool
def update_sheet_tool(task_name: str, field: str, value: str):
    """
    Updates a task in the Google Sheet. 
    Use this tool when the user asks to modify, update, change, or set a value in the tracker.
    """
    print(f"🛠 Tool Triggered: Updating {task_name}...")
    result = internal_update_task(task_name, field, value)
    return result["message"]

@tool
def send_email_tool(to_email: str, subject: str, body: str):
    """
    Sends an email using the connected SMTP server.
    You must use this tool if the user asks to 'send an email', 'notify', or 'message' someone via email.
    Do not ask for confirmation, just send it.
    """
    print(f"📧 Tool Triggered: Sending email to {to_email}...")
    result = internal_send_email(to_email, subject, body)
    return result["message"]

# --- 8. CHAT AGENT ---

@app.post("/api/chat")
def chat(request: PromptRequest):
    global excel_text_context
    
    try:
        # Reload if empty
        if not document_loaded or not excel_text_context:
            load_data_global()

        # Define Tools
        tools = [update_sheet_tool, send_email_tool]
        tool_map = {
            "update_sheet_tool": update_sheet_tool,
            "send_email_tool": send_email_tool
        }

        llm = ChatOpenAI(
            model="gpt-4o", 
            openai_api_key=openai_key,
            temperature=0
        )
        llm_with_tools = llm.bind_tools(tools)

        system_msg = f"""
        You are an advanced Project Manager Agent with REAL-WORLD CAPABILITIES.
        
        CURRENT DATA CONTEXT:
        {excel_text_context}
        
        YOUR TOOLS (You MUST use them when requested):
        1. 'update_sheet_tool': Use this to change data in the sheet.
        2. 'send_email_tool': Use this to send actual emails. **You are authorized to send emails.**
        
        INSTRUCTIONS:
        - If the user asks to "Send an email to [email]", call 'send_email_tool' immediately.
        - If the user provides a vague email request (e.g., "Email John"), check the data for an email address or ask for it.
        - If the user asks for a Chart/Table, return this specific JSON format:
        
        FORMAT FOR CHART:
        ```json
        {{ "is_chart": true, "chart_type": "bar", "title": "Tasks by Status", "data": {{ "labels": ["Done", "Pending"], "values": [5, 2] }}, "summary": "Here is the chart." }}
        ```
        """

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=request.prompt)
        ]

        print("🤖 AI Thinking...")
        ai_response = llm_with_tools.invoke(messages)

        # --- CASE A: TOOL CALLS ---
        if ai_response.tool_calls:
            print(f"🔧 AI decided to use tools: {len(ai_response.tool_calls)}")
            results = []
            
            for tool_call in ai_response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                if tool_name in tool_map:
                    print(f"   -> Executing {tool_name} with args: {tool_args}")
                    tool_output = tool_map[tool_name].invoke(tool_args)
                    results.append(tool_output)
                else:
                    results.append(f"Error: Tool {tool_name} not found.")

            return {
                "response": " | ".join(results),
                "type": "text",
                "status": "success"
            }

        # --- CASE B: JSON VISUALS ---
        content = ai_response.content.strip()
        if "```json" in content:
            try:
                clean_json = content.split("```json")[1].split("```")[0].strip()
                data_obj = json.loads(clean_json)
                
                if data_obj.get("is_chart"):
                    return {"response": data_obj["summary"], "chart_data": data_obj, "type": "chart", "status": "success"}
                
                if data_obj.get("is_table"):
                    return {"response": data_obj["summary"], "table_data": data_obj, "type": "table", "status": "success"}
            except Exception:
                pass

        # --- CASE C: TEXT ---
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
    


