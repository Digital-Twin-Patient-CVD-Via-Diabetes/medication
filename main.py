import os
import logging
import json
from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class MedicationRequest(BaseModel):
    med_name: str

class MedicationResponse(BaseModel):
    Medication: str
    specialization: str
    Influence: str

def parse_gemini_response(response_text: str) -> Dict:
    """Safely parse Gemini API response"""
    try:
        # First try to parse as JSON
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If JSON fails, try to handle as Python dict literal
        try:
            # Remove any markdown code fences if present
            cleaned = response_text.strip().replace('```json', '').replace('```', '').strip()
            return eval(cleaned)  # Note: In production, use ast.literal_eval instead
        except Exception as e:
            logger.error(f"Failed to parse response: {response_text}")
            raise ValueError(f"Could not parse API response: {str(e)}")

@app.post("/medication-info", response_model=MedicationResponse)
async def get_med_info(request: MedicationRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error - missing API key"
        )

    prompt = f"""
    Return a JSON object with medication information in this exact format:
    {{
        "Medication": "{request.med_name}",
        "specialization": "<medical specialization>",
        "Influence": "<positive/negative/neutral>"
    }}
    """
    
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            params={"key": GEMINI_API_KEY},
            json={
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            },
            timeout=15
        )
        response.raise_for_status()
        
        # Extract the text response
        data = response.json()
        response_text = data['candidates'][0]['content']['parts'][0]['text']
        
        # Parse the response
        result = parse_gemini_response(response_text)
        
        # Validate the response structure
        if not all(key in result for key in ["Medication", "specialization", "Influence"]):
            raise ValueError("Missing required fields in response")
            
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise HTTPException(status_code=502, detail="AI service unavailable")
    except ValueError as e:
        logger.error(f"Response parsing failed: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {
        "status": "running",
        "configured": bool(GEMINI_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
