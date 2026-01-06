import json
import os
import io
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set!")

# Global variables
vector_store = None
document_loaded = False

class PromptRequest(BaseModel):
    prompt: str

def load_document_on_startup():
    """Load Excel from backend folder on startup"""
    global vector_store, document_loaded
    
    try:
        # File name must match what you upload to Render
        excel_path = "ProjectPlan.xlsx" 
        
        if not os.path.exists(excel_path):
            print(f"⚠️  Warning: {excel_path} not found in project folder")
            document_loaded = False
            return
        
        print(f"✅ Reading Excel file: {excel_path}...")
        
        # Read the Excel file and handle missing values
        df = pd.read_excel(excel_path).fillna("N/A")
        
        # Ensure dates are strings to avoid timestamp issues
        if 'Start Date' in df.columns:
            df['Start Date'] = df['Start Date'].astype(str)
        if 'Finish Date' in df.columns:
            df['Finish Date'] = df['Finish Date'].astype(str)

        chunks = []
        
        # Iterate through rows and convert them to context text
        for index, row in df.iterrows():
            row_text = (
                f"Task Id: {row.get('Task Id', 'N/A')}\n"
                f"Task Name: {row.get('Task Name', 'N/A')}\n"
                f"Start Date: {row.get('Start Date', 'N/A')}\n"
                f"Finish Date: {row.get('Finish Date', 'N/A')}\n"
                f"Duration: {row.get('Duration', 'N/A')}\n"
                f"Predecessors: {row.get('Predecessors', 'N/A')}\n"
                f"Resource: {row.get('Resource', 'N/A')}"
            )
            chunks.append(row_text)

        print(f"✅ Processed {len(chunks)} rows (tasks) from Excel")
        
        # Create embeddings and vector store
        print(f"✅ Creating embeddings...")
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vector_store = FAISS.from_texts(chunks, embeddings)
        document_loaded = True
        
        print(f"✅ Document loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading document: {str(e)}")
        document_loaded = False

@app.on_event("startup")
async def startup_event():
    """Load document when server starts"""
    load_document_on_startup()

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {"status": "running", "message": "AI Project Assistant API"}

@app.get("/api/status")
def get_status():
    """Check if document is loaded"""
    return {
        "document_loaded": document_loaded,
        "status": "ready" if document_loaded else "Excel file not found"
    }

@app.post("/api/chat")
def chat(request: PromptRequest):
    """Chat with document context and support Graph Data extraction"""
    global vector_store
    
    try:
        if not document_loaded or vector_store is None:
            return {
                "response": "Excel file is not loaded. Please upload 'ProjectPlan.xlsx'.",
                "status": "error"
            }
        
        # 1. Search similar documents
        # We increase k=15 to get enough data points for a small chart
        docs = vector_store.similarity_search(request.prompt, k=15)
        context = "\n\n---\n".join([doc.page_content for doc in docs])
        
        # 2. System Instruction for JSON output
        system_instruction = """
        You are a Project Management AI Assistant. Use the provided Excel context to answer.
        
        IMPORTANT INSTRUCTIONS FOR OUTPUT FORMAT:
        
        1. **Text Answer**: If the user asks a general question (e.g., "Who is doing X?", "What is the date for Y?"), reply with a normal text explanation.
        
        2. **Chart Request**: If the user asks for a GRAPH, CHART, VISUALIZATION, or STATISTICS, you must return a **single JSON object** strictly in this format:
           {
             "is_chart": true,
             "chart_type": "bar", 
             "title": "Chart Title",
             "data": {
               "labels": ["Label A", "Label B"],
               "values": [10, 25]
             },
             "summary": "A short sentence explaining the data."
           }
           - Supported chart_types: "bar", "pie", "line".
           - "values" must be numbers.
           - Do not include markdown formatting (like ```json). Just the raw JSON string.
        """

        full_prompt = (
            f"{system_instruction}\n\n"
            f"Context Data from Excel:\n{context}\n\n"
            f"User Question: {request.prompt}\n\n"
            f"Answer:"
        )
        
        llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
        response = llm.invoke(full_prompt)
        content = response.content.strip()
        
        # 3. Clean and Parse Response
        # Sometimes AI wraps JSON in ```json ... ``` blocks. We clean that up.
        cleaned_content = content
        if cleaned_content.startswith("```"):
            cleaned_content = cleaned_content.replace("```json", "").replace("```", "").strip()

        try:
            # Try to parse the content as JSON
            data_obj = json.loads(cleaned_content)
            
            # Check if it follows our chart schema
            if isinstance(data_obj, dict) and data_obj.get("is_chart") is True:
                return {
                    "response": data_obj["summary"],
                    "chart_data": data_obj,
                    "type": "chart",
                    "status": "success"
                }
        except json.JSONDecodeError:
            # If it's not JSON, it's just a normal conversation
            pass

        # Return standard text response
        return {
            "response": content,
            "type": "text",
            "status": "success"
        }

    except Exception as e:
        return {"response": f"Internal Error: {str(e)}", "status": "error"}

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 is required for Render
    uvicorn.run(app, host="0.0.0.0", port=8000)
