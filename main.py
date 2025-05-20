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

# Load and validate environment variables
def get_required_env(var_name: str) -> str:
    """Get and validate required environment variable"""
    value = os.getenv(var_name)
    if not value:
        logger.error(f"Missing required environment variable: {var_name}")
        raise EnvironmentError(f"Missing {var_name} in environment")
    logger.info(f"Loaded environment variable: {var_name}")
    return value

# Load environment variables with validation
try:
    logger.info("Loading environment variables...")
    GEMINI_API_KEY = get_required_env("GEMINI_API_KEY")
except EnvironmentError as e:
    logger.error(f"Environment configuration error: {str(e)}")
    raise

class MedicationRequest(BaseModel):
    med_name: str

class MedicationResponse(BaseModel):
    Medication: str
    specialization: str
    Influence: str

def get_medication_info(med_name: str) -> Dict:
    """
    Fetch medication info from Gemini API and return as a dictionary
    """
    prompt_text = f"""
You are a medical-data assistant. When given the name of a medicine, you must output a Python dict literal with exactly three key-value pairs:

  "Medication": <medicine name>,
  "specialization": <specialization string>,
  "Influence": <influence string>

Always:
  1. Use double quotes for keys and string values.
  2. Do NOT output JSON or CSV or markdown fences—just the Python dict literal.

Example:
{'{' }"Medication": "Glucophage", "specialization": "Diabetes", "Influence": "Positive"{'}'}

Now output for: {med_name}
"""

    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash:generateContent"
    )
    params = {'key': GEMINI_API_KEY}
    payload = {'contents': [{'parts': [{'text': prompt_text}]}]}
    headers = {'Content-Type': 'application/json'}

    logger.info(f"Requesting medication info for: {med_name}")
    try:
        response = requests.post(
            endpoint,
            params=params,
            headers=headers,
            json=payload,
            timeout=10  # Add timeout to prevent hanging
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    
    data = response.json()

    # Extract raw dict text
    try:
        raw_text = data['candidates'][0]['content']['parts'][0]['text'].strip()
        logger.debug(f"Received raw response: {raw_text}")
    except (KeyError, IndexError) as e:
        error_msg = f"Unexpected API response format: {data}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    # Parse Python dict literal
    try:
        info_dict = ast.literal_eval(raw_text)
        logger.info(f"Successfully parsed response for {med_name}")
        return info_dict
    except Exception as e:
        error_msg = f"Failed to parse dict from model output: {raw_text}\nError: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e

@app.post("/medication-info", response_model=MedicationResponse)
async def medication_info(request: MedicationRequest):
    """
    Get medication information including specialization and influence
    
    - **med_name**: Name of the medication to look up
    """
    try:
        logger.info(f"Processing request for medication: {request.med_name}")
        info = get_medication_info(request.med_name)
        return info
    except Exception as e:
        logger.error(f"Error processing request for {request.med_name}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not retrieve medication information: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "healthy", "version": app.version}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level="info"
    )
