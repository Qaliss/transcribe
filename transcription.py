from groq import Groq
import os
import logging
import time
from fastapi import FastAPI, UploadFile, File, HTTPException

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_API_KEY = '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'
ai_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

@app.post("/transcribe/")
async def transcribe_audio(file:UploadFile = File(...)):
    start_time = time.time()
    
    try:
        audio_data = await file.read()

        transcription = ai_client.audio.transcriptions.create(
            file=(file.filename, audio_data),
            model="whisper-large-v3",
            temperature=0.0,
            response_format = "json"
        )

        elapsed_time = time.time() - start_time

        return {
            "text": transcription.text,
            "processing_time": elapsed_time,
            "model": "whisper-large-v3"
        }
    
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")
    
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Welcome to the Audio Transcription Service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
    


