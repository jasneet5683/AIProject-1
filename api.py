import json
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# NEW IMPORT FOR GOOGLE
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

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

# GET GOOGLE KEY
api_key = os.getenv("GOOGLE_API_KEY")

# Global variables
excel_text_context = ""
document_loaded = False
EXCEL_PATH = "ProjectPlan.xlsx"

# --- DATA MODELS ---

class PromptRequest(BaseModel):
    prompt: str

# Model for Adding a new task
class AddTaskRequest(BaseModel):
    task_name: str
    start_date: str
    end_date: str
    status: str
    assigned_to: str

# Model for Updating an existing task
class UpdateTaskRequest(BaseModel):
    task_name: str      # Identify task by name
    field_to_update: str # e.g., "Status", "Assigned To"
    new_value: str

# --- CORE FUNCTIONS ---

def load_excel_global():
    """
    Loads Excel and converts to CSV string.
    Refreshes the AI's memory of the project data.
    """
    global excel_text_context, document_loaded
    
    if not os.path.exists(EXCEL_PATH):
        print(f"⚠️ Warning: {EXCEL_PATH} not found.")
        document_loaded = False
        return

    try:
        print(f"✅ Reading Excel file: {EXCEL_PATH}...")
        
        # Read Excel
        df = pd.read_excel(EXCEL_PATH)
        
        # Basic cleanup
        df.dropna(how='all', inplace=True) # Drop empty rows
        df = df.fillna("N/A") # Fill blanks
        
        # Normalize Dates
        for col in df.columns:
            if "date" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                except:
                    df[col] = df[col].astype(str)

        # Convert to String for AI Context
        excel_text_context = df.to_csv(index=False)
        
        document_loaded = True
        print(f"✅ Data Loaded! ({len(df)} rows).")
        
    except Exception as e:
        print(f"❌ Error loading Excel: {str(e)}")
        document_loaded = False

@app.on_event("startup")
async def startup_event():
    load_excel_global()

# --- API ENDPOINTS ---

@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "Welcome! The backend is running. Use /api/chat, /api/add-task, or /api/update-task."
    }

@app.get("/api/status")
def get_status():
    return {"document_loaded": document_loaded}

# 1. ADD TASK ENDPOINT
@app.post("/api/add-task")
def add_task(request: AddTaskRequest):
    try:
        # Load current data
        if os.path.exists(EXCEL_PATH):
            df = pd.read_excel(EXCEL_PATH)
        else:
            return {"message": "Excel file not found", "status": "error"}

        # Create new row
        new_row = {
            "Task Name": request.task_name,
            "Start Date": request.start_date,
            "End Date": request.end_date,
            "Status": request.status,
            "Assigned To": request.assigned_to
        }

        # Add to DataFrame
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)

        # Save back to Excel
        df.to_excel(EXCEL_PATH, index=False)

        # Refresh AI Memory
        load_excel_global()

        return {"message": "Task added successfully!", "status": "success"}

    except Exception as e:
        return {"message": f"Error adding task: {str(e)}", "status": "error"}

# 2. UPDATE TASK ENDPOINT
@app.post("/api/update-task")
def update_task(request: UpdateTaskRequest):
    try:
        # Load Data
        if os.path.exists(EXCEL_PATH):
            df = pd.read_excel(EXCEL_PATH)
        else:
            return {"message": "Excel file not found", "status": "error"}

        # Find the task (Using exact name match)
        mask = df["Task Name"] == request.task_name

        if not mask.any():
            return {"message": f"Task '{request.task_name}' not found.", "status": "error"}

        # Handle Column Matching (Case insensitive)
        # Create a map: { "status": "Status", "assigned to": "Assigned To" }
        col_map = {c.strip().lower(): c for c in df.columns}
        target_col = col_map.get(request.field_to_update.strip().lower())

        if not target_col:
            return {"message": f"Column '{request.field_to_update}' not found in Excel.", "status": "error"}

        # Update the value
        df.loc[mask, target_col] = request.new_value

        # Save
        df.to_excel(EXCEL_PATH, index=False)

        # Refresh AI Memory
        load_excel_global()

        return {"message": f"Updated '{target_col}' for task '{request.task_name}'", "status": "success"}

    except Exception as e:
        return {"message": f"Error updating task: {str(e)}", "status": "error"}

# 3. CHAT ENDPOINT
@app.post("/api/chat")
def chat(request: PromptRequest):
    global excel_text_context
    
    try:
        if not document_loaded:
            return {"response": "Excel file not loaded.", "status": "error"}

        # SYSTEM INSTRUCTION
        system_instruction = f"""
        You are a Data Analyst AI. 
        Below is the raw CSV data from a Project Plan:
        
        --- DATA START ---
        {excel_text_context}
        --- DATA END ---

        INSTRUCTIONS:
        1. Analyze the data above to answer the user's question.
        2. Be precise with numbers, dates, and names.
        
        OUTPUT FORMATS (Strict JSON):
        
        A) FOR TABLES/LISTS:
           {{
             "is_table": true,
             "title": "Table Title",
             "columns": ["Col A", "Col B"],
             "rows": [ ["Val 1", "Val 2"], ["Val 3", "Val 4"] ],
             "summary": "Brief summary."
           }}

        B) FOR CHARTS/GRAPHS:
           {{
             "is_chart": true,
             "chart_type": "bar", 
             "title": "Chart Title",
             "data": {{ "labels": ["Label1", "Label2"], "values": [10, 20] }},
             "summary": "Brief summary."
           }}
           
        C) FOR TEXT:
           Return plain text.
        """

        #  INITIALIZE GOOGLE GEMINI
        #  UPDATED TO 2.5 AS REQUESTED
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  
            google_api_key=api_key,
            temperature=0  # 0 means be factual, not creative
        )
        
        full_prompt = f"{system_instruction}\n\nUser Question: {request.prompt}"
        
        # Invoke Gemini
        response = llm.invoke(full_prompt)
        content = response.content.strip()
        
        # Clean Markdown (Gemini often adds ```json ... ```)
        clean_content = content.replace("```json", "").replace("```", "").strip()

        try:
            data_obj = json.loads(clean_content)
            
            if data_obj.get("is_chart") is True:
                return {
                    "response": data_obj["summary"],
                    "chart_data": data_obj,
                    "type": "chart",
                    "status": "success"
                }
            
            if data_obj.get("is_table") is True:
                 return {
                    "response": data_obj["summary"],
                    "table_data": data_obj,
                    "type": "table",
                    "status": "success"
                }
        except:
            pass

        return {
            "response": clean_content,
            "type": "text",
            "status": "success"
        }

    except Exception as e:
        return {"response": f"Internal Error: {str(e)}", "status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
