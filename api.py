from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from PyPDF2 import PdfReader
import io
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
    """Load PDF from backend folder on startup"""
    global vector_store, document_loaded
    
    try:
        pdf_path = "WeeklyStatus_12Dec-2025.pdf"
        
        if not os.path.exists(pdf_path):
            print(f"⚠️  Warning: {pdf_path} not found in project folder")
            document_loaded = False
            return
        
        # Read PDF
        with open(pdf_path, "rb") as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        print(f"✅ Extracting text from PDF...")
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        print(f"✅ Created {len(chunks)} chunks from document")
        
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
    return {"status": "running", "message": "AI Document Assistant API"}

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
                "response": "Document is not loaded. Please add 'document.pdf' to the backend.",
                "status": "error"
            }
        
        # Search similar documents
        docs = vector_store.similarity_search(request.prompt, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Create prompt with context
        full_prompt = f"Context from document:\n{context}\n\nQuestion: {request.prompt}\n\nAnswer:"
        
        llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
        response = llm.invoke(full_prompt)
        return {"response": response.content, "status": "success"}
    except Exception as e:
        return {"response": str(e), "status": "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

