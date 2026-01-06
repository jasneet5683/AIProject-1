from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# We use pandas for Excel processing
import pandas as pd
import os
from dotenv import load_dotenv
import json 

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
        # 1. Update your file name here
        excel_path = "ProjectPlan.xlsx" 
        
        if not os.path.exists(excel_path):
            print(f"⚠️  Warning: {excel_path} not found in project folder")
            document_loaded = False
            return
        
        print(f"✅ Reading Excel file...")
        
        # 2. Read the Excel file
        # 'fillna' replaces empty cells with "N/A" to prevent errors
        df = pd.read_excel(excel_path).fillna("N/A")
        
        chunks = []
        
        # 3. Iterate through rows and convert them to structured text
        # This gives the AI context about what each value represents.
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
        
        # 4. Create embeddings and vector store
        # Note: We pass the list of row_strings directly. 
        # Since rows are usually short, we don't strictly need a TextSplitter here.
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
    return {"status": "running", "message": "AI Excel Assistant API"}

@app.get("/api/status")
def get_status():
    """Check if document is loaded"""
    return {
        "document_loaded": document_loaded,
        "status": "ready" if document_loaded else "document not found"
    }


@app.post("/api/chat")
def chat(request: PromptRequest):
    """Chat with document context and support Graph Data extraction"""
    global vector_store, document_loaded
    
    try:
        if not document_loaded or vector_store is None:
            return {
                "response": "Excel file is not loaded.",
                "status": "error"
            }
        
        # 1. Search similar documents
        docs = vector_store.similarity_search(request.prompt, k=10) # Increased k to get more data for graphs
        context = "\n\n---\n".join([doc.page_content for doc in docs])
        
        # 2. refined System Prompt
        # We tell the AI: If the user wants a chart, return JSON. If text, return text.
        system_instruction = """
        You are a Project Management AI. 
        
        RULES:
        1. If the user asks for a summary, explanation, or specific detail, answer in normal text.
        2. If the user asks for a GRAPH, CHART, or VISUALIZATION, you must return a JSON object strictly in this format:
           {
             "is_chart": true,
             "chart_type": "bar", (or "pie", "line")
             "title": "Chart Title",
             "data": {
               "labels": ["Label1", "Label2", "Label3"],
               "values": [10, 20, 30]
             },
             "summary": "A brief text summary of what the chart shows."
           }
        3. Do not add Markdown formatting (like ```json) if returning JSON. Just raw JSON string.
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
        
        # 3. Try to parse as JSON (to see if it's a graph or text)
        try:
            # If the AI decided to make a chart, this will succeed
            graph_data = json.loads(content)
            
            # If it's our specific chart format
            if graph_data.get("is_chart"):
                return {
                    "response": graph_data["summary"], 
                    "chart_data": graph_data, 
                    "type": "chart",
                    "status": "success"
                }
        except json.JSONDecodeError:
            # If it fails, it's just normal text
            pass

        # Return normal text response
        return {
            "response": content, 
            "type": "text", 
            "status": "success"
        }

    except Exception as e:
        return {"response": str(e), "status": "error"}


# @app.post("/api/chat")
# def chat(request: PromptRequest):
#    """Chat with document context"""
#    global vector_store
#    
#    try:
#        if not document_loaded or vector_store is None:
#            return {
#                "response": "Excel file is not loaded. Please add the .xlsx file to the backend.",
#                "status": "error"
#           }
#        
#        # Search similar documents (Retrieve top 5 relevant rows/tasks)
#        docs = vector_store.similarity_search(request.prompt, k=5)
#        context = "\n\n---\n".join([doc.page_content for doc in docs])
#        
#        # Create prompt with context
#        full_prompt = (
#            f"You are a Project Management AI helper. Use the following project tasks to answer the question.\n\n"
#            f"Context Data:\n{context}\n\n"
#            f"Question: {request.prompt}\n\n"
#            f"Answer:"
#        )
#        
#        llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
#        response = llm.invoke(full_prompt)
#        return {"response": response.content, "status": "success"}
#    except Exception as e:
#        return {"response": str(e), "status": "error"}
#
if __name__ == "__main__":
    import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)
