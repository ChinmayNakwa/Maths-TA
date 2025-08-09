# backend/utils/voice_service.py

import requests
from ..config import settings
from typing import Iterator
from huggingface_hub import InferenceClient
import time

HF_API_URL_STT = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
HF_AUTH_HEADERS = {"Authorization": f"Bearer {settings.HF_TOKEN}"}

print("VoiceService: Initializing Fal AI TTS client...")
try:
    # We use the Hugging Face token for authentication with the provider
    tts_client = InferenceClient(
        provider="fal-ai",
        token=settings.HF_TOKEN,
    )
    print("VoiceService: Fal AI TTS client initialized successfully.")
    TTS_CLIENT_INITIALIZED = True
except Exception as e:
    print(f"FATAL: Could not initialize Fal AI TTS client. Error: {e}")
    TTS_CLIENT_INITIALIZED = False


def transcribe_audio(audio_data: bytes) -> str:
    """
    Sends audio data to the Hugging Face Whisper API for transcription.
    (This function is unchanged).
    """
    print("VoiceService: Transcribing audio with Whisper API...")
    try:
        headers = {**HF_AUTH_HEADERS, "Content-Type": "audio/webm"}
        response = requests.post(HF_API_URL_STT, headers=headers, data=audio_data)
        response.raise_for_status()
        result = response.json()
        return result.get("text", "").strip()
    except Exception as e:
        print(f"Whisper API Error: {e}")
        return "Sorry, there was an error understanding speech."


def text_to_speech(text: str) -> Iterator[bytes]:
    """
    Generates speech using the Fal AI provider via the InferenceClient.
    """
    if not TTS_CLIENT_INITIALIZED:
        print("VoiceService: TTS client not initialized, cannot generate speech.")
        yield b""
        return

    print(f"VoiceService: Synthesizing speech via Fal AI for text: '{text}'")
    
    # Retry logic for potential transient network issues
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # The client returns the raw audio bytes directly
            audio_bytes = tts_client.text_to_speech(
                text,
                model="hexgrad/Kokoro-82M", # Using a high-quality, fast model on Fal
            )
            
            print("VoiceService: Speech synthesis successful. Streaming to client.")
            chunk_size = 4096
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i+chunk_size]
            
            return 

        except Exception as e:
            print(f"VoiceService: Error calling Fal AI TTS (Attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2) # Wait 2 seconds before retrying
            else:
                print("VoiceService: All TTS attempts failed.")
                yield b"" # Yield empty bytes on final failure