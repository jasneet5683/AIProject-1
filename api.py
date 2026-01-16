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

# --- AI TOOLS DEFINITION ---

@tool
def update_sheet_tool(task_name: str, field: str, value: str):
    """Updates a task in the Google Sheet."""
    print(f"🛠 Tool: Updating Sheet -> {task_name}")
    result = internal_update_task(task_name, field, value)
    return result["message"]

@tool
def send_email_tool(to_email: str, subject: str, body: str):
    """
    Sends an email to a specific address.
    Args:
        to_email: The recipient's email address (e.g. 'client@example.com').
        subject: The subject line of the email.
        body: The main content/message of the email.
    """
    print(f"📧 Tool: Sending Email -> {to_email}")
    result = internal_send_email(to_email, subject, body)
    return result["message"]

# --- CHAT ENDPOINT ---

@app.post("/api/chat")
def chat(request: PromptRequest):
    global excel_text_context
    
    try:
        if not document_loaded: load_data_global()

        # DEFINE TOOLS (Now includes Email!)
        tools = [update_sheet_tool, send_email_tool]
        
        llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_key, temperature=0)
        llm_with_tools = llm.bind_tools(tools)

        system_msg = f"""
        You are an advanced Project Manager Assistant.
        
        CURRENT DATA:
        {excel_text_context}
        
        INSTRUCTIONS:
        1. **UPDATE DATA**: If asked to change data, use 'update_sheet_tool'.
        2. **SEND EMAIL**: If asked to email someone, use 'send_email_tool'. 
           - If the user says "Email John", look for John's email in the data or ask for it.
           - Write a professional subject and body if not provided.
        3. **VISUALS**: For Charts/Tables, return the JSON format as defined previously.
        
        FORMAT FOR CHARTS (Only if requested):
        ```json
        {{ "is_chart": true, "chart_type": "bar", "title": "...", "data": {{...}}, "summary": "..." }}
        ```
        """

        messages = [SystemMessage(content=system_msg), HumanMessage(content=request.prompt)]

        print("🤖 AI Thinking...")
        ai_response = llm_with_tools.invoke(messages)

        # CHECK FOR TOOL CALLS
        if ai_response.tool_calls:
            results = []
            for tool_call in ai_response.tool_calls:
                # Select the right tool
                if tool_call["name"] == "update_sheet_tool":
                    res = update_sheet_tool.invoke(tool_call["args"])
                elif tool_call["name"] == "send_email_tool":
                    res = send_email_tool.invoke(tool_call["args"])
                else:
                    res = "Unknown Tool"
                results.append(res)
            
            return {"response": " | ".join(results), "type": "text", "status": "success"}

        # CHECK FOR JSON VISUALS
        content = ai_response.content.strip()
        if "```json" in content:
            try:
                clean_json = content.split("```json")[1].split("```")[0].strip()
                data_obj = json.loads(clean_json)
                if data_obj.get("is_chart"):
                    return {"response": data_obj["summary"], "chart_data": data_obj, "type": "chart", "status": "success"}
                if data_obj.get("is_table"):
                    return {"response": data_obj["summary"], "table_data": data_obj, "type": "table", "status": "success"}
            except:
                pass

        return {"response": content, "type": "text", "status": "success"}

    except Exception as e:
        return {"response": f"Error: {str(e)}", "status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
