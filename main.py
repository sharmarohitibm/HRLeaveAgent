from fastapi import FastAPI
from pydantic import BaseModel
from apply_leave_crew_sqlite import queryleave
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/leave-balance")
async def get_leave_balance(req: QueryRequest):
    try:
        response = queryleave(req.question)
        if isinstance(response, dict) and "raw" in response:
            return response["raw"]
        elif hasattr(response, "get") and response.get("raw"):
            return response.get("raw")
        return response
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
