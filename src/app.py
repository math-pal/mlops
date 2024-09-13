import sys  #, os
# project_root = "E:/Training/Atomcamp/DS6_Bootcamp/Sessions/Guiede_Projects/mlops"
# src_path = os.path.join(project_root, "src")
# sys.path.insert(0, src_path)  # Ensure 'src' is in the Python path
# Or, use the following
sys.path.append('E:/Training/Atomcamp/DS6_Bootcamp/Sessions/Guiede_Projects/mlops')
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.main import retriever

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(query: QueryRequest):
    # Validate the input
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Call the main logic
    answer = retriever(query.question)
    
    return {"question": query.question, "answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)  # , host="0.0.0.0", port=8000