from groq import Groq
import os
import logging
import time
from fastapi import FastAPI, Request

print("🔵 FastAPI startup: file executed")
logging.basicConfig(level=logging.INFO)
logging.info("🔵 Logging initialized")


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ai_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(redirect_slashes=False)

@app.on_event("startup")
async def startup_event():
    print("🟢 FastAPI fully started")


@app.post("/transcribe")
async def transcribe_audio(request: Request):
    start_time = time.time()
    
    try:
        audio_data = await request.body()

        if len(audio_data) == 0:
            return {"error": "No audio data provided"}
        
        transcription = ai_client.audio.transcriptions.create(
            file=("audio.wav", audio_data),
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
        return {"error": "Transcription failed"}
    
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def root():
    logging.info("🟣 Root route hit")
    return {"message": "Welcome to the Audio Transcription Service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
    


