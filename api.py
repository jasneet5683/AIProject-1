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
    """Chat with document context"""
    global vector_store
    
    try:
        if not document_loaded or vector_store is None:
            return {
                "response": "Excel file is not loaded. Please add the .xlsx file to the backend.",
                "status": "error"
            }
        
        # Search similar documents (Retrieve top 5 relevant rows/tasks)
        docs = vector_store.similarity_search(request.prompt, k=5)
        context = "\n\n---\n".join([doc.page_content for doc in docs])
        
        # Create prompt with context
        full_prompt = (
            f"You are a Project Management AI helper. Use the following project tasks to answer the question.\n\n"
            f"Context Data:\n{context}\n\n"
            f"Question: {request.prompt}\n\n"
            f"Answer:"
        )
        
        llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
        response = llm.invoke(full_prompt)
        return {"response": response.content, "status": "success"}
    except Exception as e:
        return {"response": str(e), "status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
