import json
import os
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
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

api_key = os.getenv("OPENAI_API_KEY")

# Global variables
excel_text_context = ""
document_loaded = False

class PromptRequest(BaseModel):
    prompt: str

def load_excel_global():
    """
    Loads the entire Excel file into a text string.
    This ensures the AI sees ALL data, not just bits and pieces.
    """
    global excel_text_context, document_loaded
    
    excel_path = "ProjectPlan.xlsx" 
    
    if not os.path.exists(excel_path):
        print(f"⚠️ Warning: {excel_path} not found.")
        document_loaded = False
        return

    try:
        print(f"✅ Reading Excel file: {excel_path}...")
        
        # Read Excel
        df = pd.read_excel(excel_path).fillna("N/A")
        
        # Convert Dates to string to prevent format errors
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                df[col] = df[col].astype(str)

        # Convert the ENTIRE dataframe to a CSV-style string
        # This gives the AI a perfect view of the data structure
        excel_text_context = df.to_csv(index=False)
        
        document_loaded = True
        print(f"✅ Data Loaded! ({len(df)} rows)")
        
    except Exception as e:
        print(f"❌ Error loading Excel: {str(e)}")
        document_loaded = False

@app.on_event("startup")
async def startup_event():
    load_excel_global()

@app.get("/")
def read_root():
    return {"status": "running"}

@app.get("/api/status")
def get_status():
    return {"document_loaded": document_loaded}

@app.post("/api/chat")
def chat(request: PromptRequest):
    global excel_text_context
    
    try:
        if not document_loaded:
            return {"response": "Excel file not loaded.", "status": "error"}

        # SYSTEM INSTRUCTION
        # We pass the 'excel_text_context' directly into the prompt.
        system_instruction = f"""
        You are an expert Project Management AI. 
        Below is the COMPLETE DATA from the Project Plan Excel file:
        
        --- START OF DATA ---
        {excel_text_context}
        --- END OF DATA ---

        INSTRUCTIONS:
        1. Answer based ONLY on the data above. Be accurate.
        2. If asked for a specific value (date, duration), look it up exactly.
        
        OUTPUT FORMATS:
        
        A) If user asks for a TABLE/LIST:
           Return ONLY this raw JSON (no markdown):
           {{
             "is_table": true,
             "title": "Title Here",
             "columns": ["Col 1", "Col 2"],
             "rows": [ ["Val 1", "Val 2"], ["Val 3", "Val 4"] ],
             "summary": "Short summary."
           }}

        B) If user asks for a CHART/GRAPH:
           Return ONLY this raw JSON (no markdown):
           {{
             "is_chart": true,
             "chart_type": "bar",
             "title": "Chart Title",
             "data": {{ "labels": ["A", "B"], "values": [10, 20] }},
             "summary": "Short summary."
           }}
           
        C) Otherwise, return plain text.
        """

        # Call OpenAI
        llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0)
        
        # Combine instructions + user question
        full_prompt = f"{system_instruction}\n\nUser Question: {request.prompt}"
        
        response = llm.invoke(full_prompt)
        content = response.content.strip()
        
        # Clean potential Markdown formatting
        clean_content = content.replace("```json", "").replace("```", "").strip()

        # Try to parse JSON for Tables/Charts
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
            # Not JSON, just normal text
            pass

        return {
            "response": content,
            "type": "text",
            "status": "success"
        }

    except Exception as e:
        return {"response": f"Internal Error: {str(e)}", "status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
