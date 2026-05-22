
import os, time, hashlib, json, pathlib, random
from typing import List, Optional
from dotenv import load_dotenv
import requests

load_dotenv()
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "")
ELEVEN_VOICE_ID = os.getenv("ELEVEN_VOICE_ID", "")
BASE = "https://api.elevenlabs.io/v1"

def _headers():
    return {"xi-api-key": ELEVEN_API_KEY, "accept": "audio/mpeg", "Content-Type": "application/json"}

def generate_tts_dataset(texts: List[str], voice_id: Optional[str]=None, out_dir: str="data/raw/ai", model_id: str="eleven_monolingual_v1"):
    """Generate AI speech MP3s from ElevenLabs into out_dir. Convert to WAV (16k mono) for training."""
    voice_id = voice_id or ELEVEN_VOICE_ID
    assert ELEVEN_API_KEY, "Set ELEVEN_API_KEY in .env"
    assert voice_id, "Provide ELEVEN_VOICE_ID in .env or pass voice_id"
    os.makedirs(out_dir, exist_ok=True)
    for i, txt in enumerate(texts):
        payload = {"text": txt, "model_id": model_id, "voice_settings": {"stability": 0.4, "similarity_boost": 0.7}}
        url = f"{BASE}/text-to-speech/{voice_id}"
        r = requests.post(url, headers=_headers(), json=payload)
        if r.status_code != 200:
            print("TTS error", r.status_code, r.text[:200]); continue
        mp3_path = os.path.join(out_dir, f"elab_{i:04d}.mp3")
        with open(mp3_path, "wb") as f:
            f.write(r.content)
        print("saved", mp3_path)
    print("Done. Convert MP3 to WAV (16kHz mono) before training.")

def check_ai_speech(audio_bytes: bytes) -> dict:
    """Stub: if your plan exposes classifier API, call it here; else returns unsupported."""
    return {"supported": False, "prob_ai": None, "provider": "elevenlabs", "note": "Classifier not enabled in this template."}

# if __name__ == "__main__":
#     generate_tts_dataset()