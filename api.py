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

#  GET GOOGLE KEY
api_key = os.getenv("GOOGLE_API_KEY")

# Global variables
excel_text_context = ""
document_loaded = False

class PromptRequest(BaseModel):
    prompt: str

def load_excel_global():
    """
    Loads Excel and converts to CSV string.
    Gemini has a huge context window, so we don't need to be as aggressive
    with cutting data, but cleaning it is still good practice.
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
        df = pd.read_excel(excel_path)
        
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

        # Convert to String
        excel_text_context = df.to_csv(index=False)
        
        document_loaded = True
        print(f"✅ Data Loaded! ({len(df)} rows). Using Google Gemini.")
        
    except Exception as e:
        print(f"❌ Error loading Excel: {str(e)}")
        document_loaded = False

@app.on_event("startup")
async def startup_event():
    load_excel_global()

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
        # gemini-1.5-flash is fast, cheap, and has 1M token context
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
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
