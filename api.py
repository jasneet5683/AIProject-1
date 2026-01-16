import json
import os
import smtplib
from email.message import EmailMessage
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API KEYS
openai_key = os.getenv("OPENAI_API_KEY")
email_sender = os.getenv("EMAIL_SENDER")
email_password = os.getenv("EMAIL_PASSWORD")

# Global variables
excel_text_context = ""
document_loaded = False
SHEET_NAME = "Task_Manager"

class PromptRequest(BaseModel):
    prompt: str

# --- HELPER: CONNECT TO GOOGLE SHEETS ---
def get_google_sheet():
    try:
        json_creds = os.getenv("GOOGLE_CREDS")
        if not json_creds:
            return None
        creds_dict = json.loads(json_creds)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open(SHEET_NAME).sheet1
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return None

# --- HELPER: SEND EMAIL FUNCTION ---
def internal_send_email(to_email, subject, body):
    try:
        if not email_sender or not email_password:
            return {"message": "Email credentials missing in .env", "status": "error"}

        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = email_sender
        msg['To'] = to_email

        # Connect to Gmail SMTP
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_sender, email_password)
            smtp.send_message(msg)
        
        return {"message": f"✅ Email sent to {to_email}!", "status": "success"}
    except Exception as e:
        return {"message": f"❌ Error sending email: {str(e)}", "status": "error"}

# --- DATA LOADING ---
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
        excel_text_context = df.to_csv(index=False)
        document_loaded = True
        print("✅ Data Loaded.")
    except Exception:
        document_loaded = False

@app.on_event("startup")
async def startup_event():
    load_data_global()

# --- SHEET UPDATE LOGIC ---
def internal_update_task(task_name, field, value):
    sheet = get_google_sheet()
    if not sheet: return {"message": "Connection Error", "status": "error"}

    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Robust Column Search
        col_map = {c.strip().lower().replace("_", " "): c for c in df.columns}
        
        task_col_actual = col_map.get("task name") or col_map.get("taskname") or col_map.get("task")
        if not task_col_actual: return {"message": "Task Column not found", "status": "error"}

        target_col_clean = field.strip().lower().replace("_", " ")
        target_col_actual = col_map.get(target_col_clean)
        if not target_col_actual: return {"message": f"Column '{field}' not found", "status": "error"}

        mask = df[task_col_actual].astype(str).str.strip().str.lower() == task_name.strip().lower()
        if not mask.any(): return {"message": f"Task '{task_name}' not found", "status": "error"}

        df.loc[mask, target_col_actual] = value
        
        sheet.clear()
        sheet.update([df.columns.values.tolist()] + df.values.tolist())
        load_data_global()
        return {"message": f"✅ Updated '{task_name}' ({target_col_actual} -> {value})", "status": "success"}

    except Exception as e:
        return {"message": f"Error updating: {e}", "status": "error"}

# --- 3. SMART CHAT AGENT (TOOLS + CHARTS) ---

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

@app.post("/api/chat")
def chat(request: PromptRequest):
    global excel_text_context
    
    try:
        if not document_loaded:
            load_data_global()

        # 1. Bind Tools
        tools = [update_sheet_tool, send_email_tool]
        
        # 2. Create Tool Map (for execution later)
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

        # 3. System Prompt - EXPLICIT PERMISSIONS
        system_msg = f"""
        You are an advanced Project Manager Agent with REAL-WORLD CAPABILITIES.
        
        CURRENT DATA CONTEXT:
        {excel_text_context}
        
        YOUR TOOLS (You MUST use them when requested):
        1. 'update_sheet_tool': Use this to change data in the sheet.
        2. 'send_email_tool': Use this to send actual emails. **You are authorized to send emails.** Do not say you cannot do it.
        
        INSTRUCTIONS:
        - If the user asks to "Send an email to [email]", call 'send_email_tool' immediately.
        - If the user provides a vague email request (e.g., "Email John"), check the data for an email address or ask for it.
        - If the user asks for a Chart/Table, return the specific JSON format.
        """

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=request.prompt)
        ]

        print("🤖 AI Thinking...")
        ai_response = llm_with_tools.invoke(messages)

        # --- CASE A: TOOL CALLS (Update OR Email) ---
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
                "response": " | ".join(results), # Combine outputs if multiple tools used
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

