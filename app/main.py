from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import BarkProcessor, BarkModel
from scipy.io import wavfile
import numpy as np
import os
import logging
from datetime import datetime
import warnings

# Suppress Bark warnings if they don't affect output
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load processor and model (loaded once at startup)
processor = BarkProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small").to("cpu")


# Set a distinct pad token ID to avoid warnings
processor.pad_token_id = 1
model.config.pad_token_id = 1

# Define input data model
class AudioRequest(BaseModel):
    text: str
    voice_preset: str = "v2/en_speaker_6"
    sampling_rate: int = 24000

@app.get("/")
async def root():
    return {"message": "Welcome to the Text-to-Speech APIs. Use POST /generate-audio to generate audio."}

@app.post("/generate-audio", response_class=FileResponse)
async def generate_audio(request: AudioRequest):
    output_dir = os.getenv("OUTPUT_DIR", "./")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"brk1_{timestamp}.wav")
    try:
        logger.info(f"Processing text: {request.text} with voice_preset: {request.voice_preset}")

        # Process input text
        inputs = processor(
            request.text,
            voice_preset=request.voice_preset,
            return_tensors="pt",
            return_attention_mask=True
        )
        logger.info(f"Inputs generated: {inputs.keys()}")

        # Generate audio
        speech = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=True
        )
        logger.info(f"Audio generated, shape: {speech.shape}")

        # Process audio
        audio = speech[0].cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()
        logger.info(f"Audio processed, shape: {audio.shape}")

        # Save audio
        wavfile.write(output_file, rate=request.sampling_rate, data=audio)
        logger.info(f"Audio saved to: {output_file}")

        # Verify file
        if not os.path.exists(output_file):
            logger.error(f"File {output_file} was not created")
            raise HTTPException(status_code=500, detail="Failed to create audio file")

        return FileResponse(
            path=output_file,
            filename=f"generated_audio_{timestamp}.wav",
            media_type="audio/wav"
        )

    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")