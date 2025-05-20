import os
import logging
import ast
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medication Information API",
    description="API to get medication specialization and influence information",
    version="1.0.0",
)

def get_required_env(var_name: str) -> str:
    """Get and validate required environment variable"""
    value = os.getenv(var_name)
    if not value:
        logger.error(f"Missing required environment variable: {var_name}")
        raise EnvironmentError(f"Missing {var_name} in environment")
    logger.info(f"Loaded environment variable: {var_name}")
    return value

# Load config
try:
    GEMINI_API_KEY = get_required_env("GEMINI_API_KEY")
except EnvironmentError as e:
    logger.error(f"Configuration error: {str(e)}")
    raise

class MedicationRequest(BaseModel):
    med_name: str

class MedicationResponse(BaseModel):
    Medication: str
    specialization: str
    Influence: str

def get_medication_info(med_name: str) -> Dict:
    prompt_text = f"""..."""  # (keep your existing prompt text here)

    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    try:
        response = requests.post(
            endpoint,
            params={'key': GEMINI_API_KEY},
            headers={'Content-Type': 'application/json'},
            json={'contents': [{'parts': [{'text': prompt_text}]}]},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        raw_text = data['candidates'][0]['content']['parts'][0]['text'].strip()
        return ast.literal_eval(raw_text)
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/medication-info", response_model=MedicationResponse)
async def get_med_info(request: MedicationRequest):
    try:
        return get_medication_info(request.med_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))