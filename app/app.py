
# import os, io, base64, json
# import numpy as np
# import gradio as gr
# from dotenv import load_dotenv
# from inference import Detector
# from elevenlabs_tools import check_ai_speech

# load_dotenv()
# WEIGHTS = os.getenv("MODEL_WEIGHTS_PATH", "app/models/weights/cnn_melspec.pth")
# det = Detector(weights_path=WEIGHTS)

# def predict_and_explain(audio):
#     if isinstance(audio, tuple):
#         sr, data = audio
#         if data.ndim > 1:
#             data = data.mean(axis=1)
#         import soundfile as sf, io as _io
#         bio = _io.BytesIO()
#         sf.write(bio, data, sr, format="WAV")
#         wav_bytes = bio.getvalue()
#     else:
#         with open(audio, "rb") as f:
#             wav_bytes = f.read()
#     proba = det.predict_proba(wav_bytes)
#     exp = det.explain(wav_bytes)
#     cam = np.array(exp["cam"])
#     return proba, cam

# def provenance(audio):
#     if isinstance(audio, tuple):
#         sr, data = audio
#         if data.ndim > 1:
#             data = data.mean(axis=1)
#         import soundfile as sf, io as _io
#         bio = _io.BytesIO()
#         sf.write(bio, data, sr, format="WAV")
#         wav_bytes = bio.getvalue()
#     else:
#         with open(audio, "rb") as f:
#             wav_bytes = f.read()
#     return check_ai_speech(wav_bytes)

# with gr.Blocks(title="AI Voice Detector") as demo:
#     gr.Markdown("# 🔎 AI Voice Detector — Human vs AI Speech")
#     gr.Markdown("Upload or record a short clip (~3s). Get a probability and an explanation heatmap.")
#     with gr.Row():
#         audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio")
#         with gr.Column():
#             btn_predict = gr.Button("Analyze", variant="primary")
#             btn_prov = gr.Button("Provenance Check (optional)")
#     with gr.Row():
#         json_out = gr.JSON(label="Prediction (probabilities)")
#         #cam_out = gr.Heatmap(label="Explanation Heatmap (spectrogram importance)")
#         cam_out = gr.Image(label="Explanation Heatmap (spectrogram importance)", type="pil")
#     prov_out = gr.JSON(label="Provenance Result (if available)")
#     btn_predict.click(predict_and_explain, inputs=audio_in, outputs=[json_out, cam_out])
#     btn_prov.click(provenance, inputs=audio_in, outputs=prov_out)

# if __name__ == "__main__":
#     demo.launch()


#-----------------------------------------------------------

# import os, io, base64, json
# import numpy as np
# import gradio as gr
# from dotenv import load_dotenv
# from inference import Detector
# from elevenlabs_tools import check_ai_speech

# # NEW for color heatmap
# from matplotlib import cm

# load_dotenv()
# WEIGHTS = os.getenv("MODEL_WEIGHTS_PATH", "app/models/weights/cnn_melspec.pth")
# det = Detector(weights_path=WEIGHTS)

# def _to_wav_bytes(audio):
#     if audio is None:
#         return None
#     if isinstance(audio, tuple):
#         sr, data = audio
#         if data.ndim > 1:
#             data = data.mean(axis=1)
#         import soundfile as sf, io as _io
#         bio = _io.BytesIO()
#         sf.write(bio, data, sr, format="WAV")
#         return bio.getvalue()
#     else:
#         with open(audio, "rb") as f:
#             return f.read()

# def predict_and_explain(audio):
#     wav_bytes = _to_wav_bytes(audio)
#     if wav_bytes is None:
#         return {"error": "No audio received. Record or upload a clip."}, None

#     proba = det.predict_proba(wav_bytes)
#     exp = det.explain(wav_bytes)

#     # make a pretty color heatmap image
#     cam = np.array(exp["cam"], dtype=np.float32)
#     cam = np.clip(cam, 0.0, 1.0)
#     cam_rgb = (cm.magma(cam)[..., :3] * 255).astype(np.uint8)  # HxWx3 uint8

#     # include the predicted label in the JSON
#     return proba, cam_rgb

# def provenance(audio):
#     wav_bytes = _to_wav_bytes(audio)
#     if wav_bytes is None:
#         return {"error": "No audio received."}
#     return check_ai_speech(wav_bytes)

# with gr.Blocks(title="AI Voice Detector") as demo:
#     gr.Markdown("# 🔎 AI Voice Detector — Human vs AI Speech")
#     gr.Markdown("Upload or record a short clip (~3s). Get a probability, label, and heatmap.")
#     with gr.Row():
#         audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio")
#         with gr.Column():
#             btn_predict = gr.Button("Analyze", variant="primary")
#             btn_prov = gr.Button("Provenance Check (optional)")
#     with gr.Row():
#         json_out = gr.JSON(label="Prediction (probabilities + label)")
#         cam_out = gr.Image(label="Explanation Heatmap (spectrogram importance)")
#     prov_out = gr.JSON(label="Provenance Result (if available)")
#     btn_predict.click(predict_and_explain, inputs=audio_in, outputs=[json_out, cam_out])
#     btn_prov.click(provenance, inputs=audio_in, outputs=prov_out)

# if __name__ == "__main__":
#     demo.launch()

#-----------------------------------------------------------

import os, io
import numpy as np
import gradio as gr
from dotenv import load_dotenv
from matplotlib import cm

# Robust imports (works with: `python -m app.app` or `python app/app.py`)
try:
    from .inference import Detector
    from .elevenlabs_tools import check_ai_speech
except ImportError:
    from app.inference import Detector
    from app.elevenlabs_tools import check_ai_speech

load_dotenv()
WEIGHTS = os.getenv("MODEL_WEIGHTS_PATH", "app/models/weights/cnn_melspec.pth")
det = Detector(weights_path=WEIGHTS)

def _to_wav_bytes(audio):
    """Accepts Gradio (sr, data) or filepath string; returns WAV bytes or None."""
    if audio is None:
        return None
    if isinstance(audio, tuple):
        sr, data = audio
        if data is None:
            return None
        if data.ndim > 1:
            data = data.mean(axis=1)
        import soundfile as sf
        bio = io.BytesIO()
        sf.write(bio, data, int(sr), format="WAV")
        return bio.getvalue()
    else:
        try:
            with open(audio, "rb") as f:
                return f.read()
        except Exception:
            return None

def predict_and_explain(audio):
    wav_bytes = _to_wav_bytes(audio)
    if not wav_bytes:
        return {"error": "No audio received. Record or upload a 2–4s clip."}, None

    proba = det.predict_proba(wav_bytes)
    exp = det.explain(wav_bytes)

    # Color heatmap image
    cam = np.array(exp["cam"], dtype=np.float32)
    cam = np.clip(cam, 0.0, 1.0)
    cam_rgb = (cm.magma(cam)[..., :3] * 255).astype(np.uint8)  # HxWx3

    return proba, cam_rgb

def provenance(audio):
    wav_bytes = _to_wav_bytes(audio)
    if not wav_bytes:
        return {"error": "No audio received."}
    return check_ai_speech(wav_bytes)

with gr.Blocks(title="AI Voice Detector") as demo:
    gr.Markdown("# 🔎 AI Voice Detector — Human vs AI Speech")
    gr.Markdown("Record or upload a short clip (~3s). Get probabilities, a label, and an explanation heatmap.")
    with gr.Row():
        audio_in = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Audio")
        with gr.Column():
            btn_predict = gr.Button("Analyze", variant="primary")
            btn_prov = gr.Button("Provenance Check (optional)")
    with gr.Row():
        json_out = gr.JSON(label="Prediction (probabilities + label)")
        cam_out = gr.Image(label="Explanation Heatmap (spectrogram importance)")
    prov_out = gr.JSON(label="Provenance Result (if available)")

    btn_predict.click(predict_and_explain, inputs=audio_in, outputs=[json_out, cam_out])
    btn_prov.click(provenance, inputs=audio_in, outputs=prov_out)

if __name__ == "__main__":
    demo.launch()