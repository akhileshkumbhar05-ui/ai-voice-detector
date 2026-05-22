
# import os
# import numpy as np
# import torch
# import torch.nn.functional as F
# from models.cnn_melspec import TinyMelCNN
# from utils.audio import load_audio, pad_or_trim, logmel, heuristic_features, TARGET_SR
# from utils.gradcam import SpectrogramGradCAM

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# class Detector:
#     def __init__(self, weights_path: str | None = None, use_cuda: bool | None = None):
#         self.device = DEVICE if use_cuda is None else ("cuda" if use_cuda else "cpu")
#         self.model = TinyMelCNN().to(self.device)
#         self.trained = False
#         if weights_path and os.path.exists(weights_path):
#             state = torch.load(weights_path, map_location=self.device)
#             self.model.load_state_dict(state)
#             self.model.eval()
#             self.trained = True

#     @torch.inference_mode()
#     def predict_proba(self, wav_bytes_or_path) -> dict:
#         y, sr = load_audio(wav_bytes_or_path, target_sr=TARGET_SR)
#         y = pad_or_trim(y, duration_s=3.0, sr=sr)
#         mel = logmel(y, sr)
#         x = torch.from_numpy(mel[None, None, :, :]).to(self.device)
#         logits = self.model(x)
#         probs = F.softmax(logits, dim=-1).cpu().numpy()[0].tolist()
#         if not self.trained:
#             feats = heuristic_features(y, sr)
#             flatness = float(feats[2]); centroid = float(feats[1]); zcr = float(feats[0])
#             score = 0.5*flatness + 0.000001*centroid + 0.25*zcr
#             score = min(max(score, 0.0), 1.0)
#             ai_prob = float(0.6*score + 0.4*probs[1])
#             probs = [1.0 - ai_prob, ai_prob]
#         return {"human": float(probs[0]), "ai": float(probs[1])}

#     def explain(self, wav_bytes_or_path) -> dict:
#         y, sr = load_audio(wav_bytes_or_path, target_sr=TARGET_SR)
#         y = pad_or_trim(y, duration_s=3.0, sr=sr)
#         mel = logmel(y, sr)
#         x = torch.from_numpy(mel[None, None, :, :]).to(self.device).requires_grad_(True)
#         cam_util = SpectrogramGradCAM(self.model, target_layer_name="features.6")
#         cam, logits = cam_util(x, class_idx=None)
#         probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()[0].tolist()
#         return {"cam": cam[0].tolist(), "probs": {"human": probs[0], "ai": probs[1]}}


#----------------------------------------------------------------------------------

# app/inference.py
# import os
# import numpy as np
# import torch
# import torch.nn.functional as F

# from models.cnn_melspec import TinyMelCNN
# from utils.audio import load_audio, pad_or_trim, logmel, heuristic_features, TARGET_SR
# from utils.gradcam import SpectrogramGradCAM

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # --- knobs (set via env) ---
# USE_HEURISTIC = os.getenv("DETECTOR_ALLOW_HEURISTIC", "0") == "1"  # default OFF
# AI_THRESHOLD  = float(os.getenv("DETECTOR_AI_THRESHOLD", "0.50"))  # decision threshold
# AI_PROB_BIAS  = float(os.getenv("DETECTOR_AI_BIAS", "0.00"))       # e.g., 0.03 to lean AI slightly

# class Detector:
#     def __init__(self, weights_path: str | None = None, use_cuda: bool | None = None):
#         self.device = DEVICE if use_cuda is None else ("cuda" if use_cuda else "cpu")
#         self.model = TinyMelCNN().to(self.device)
#         self.trained = False
#         if weights_path and os.path.exists(weights_path):
#             state = torch.load(weights_path, map_location=self.device)
#             self.model.load_state_dict(state)
#             self.model.eval()
#             self.trained = True

#     @torch.inference_mode()
#     def predict_proba(self, wav_bytes_or_path) -> dict:
#         # audio -> log-mel
#         y, sr = load_audio(wav_bytes_or_path, target_sr=TARGET_SR)
#         y = pad_or_trim(y, duration_s=3.0, sr=sr)
#         mel = logmel(y, sr)
#         x = torch.from_numpy(mel[None, None]).to(self.device)

#         logits = self.model(x)
#         probs = F.softmax(logits, dim=-1).cpu().numpy()[0]  # [human, ai]

#         # Use heuristic ONLY if explicitly allowed AND no trained weights
#         if (not self.trained) and USE_HEURISTIC:
#             feats = heuristic_features(y, sr)               # [zcr, centroid, flatness, ...]
#             zcr, centroid, flatness = float(feats[0]), float(feats[1]), float(feats[2])
#             score = 0.5*flatness + 1e-6*centroid + 0.25*zcr
#             score = np.clip(score, 0.0, 1.0)
#             ai_prob = float(0.4*score + 0.6*probs[1])       # lighter mix than before
#             probs = np.array([1.0 - ai_prob, ai_prob], dtype=np.float32)

#         # Optional tiny bias toward AI to reduce false "human" on replayed TTS
#         probs[1] = float(np.clip(probs[1] + AI_PROB_BIAS, 0.0, 1.0))
#         probs[0] = float(1.0 - probs[1])

#         label = "ai" if probs[1] >= AI_THRESHOLD else "human"
#         return {
#             "human": float(probs[0]),
#             "ai": float(probs[1]),
#             "label": label,
#             "threshold": AI_THRESHOLD,
#             "trained": self.trained
#         }

#     @torch.inference_mode()
#     # def explain(self, wav_bytes_or_path) -> dict:
#     #     y, sr = load_audio(wav_bytes_or_path, target_sr=TARGET_SR)
#     #     y = pad_or_trim(y, duration_s=3.0, sr=sr)
#     #     mel = logmel(y, sr)
#     #     x = torch.from_numpy(mel[None, None]).to(self.device).requires_grad_(True)
#     #     cam_util = SpectrogramGradCAM(self.model, target_layer_name="features.6")
#     #     cam, logits = cam_util(x, class_idx=None)           # cam shape: [1, H, W]
#     #     probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()[0]
#     #     return {"cam": cam[0].tolist(), "probs": {"human": float(probs[0]), "ai": float(probs[1])}}
#     # def explain(self, wav_bytes_or_path) -> dict:
#     #     # DO NOT decorate this function with @torch.inference_mode
#     #     self.model.eval()

#     #     # build input
#     #     y, sr = load_audio(wav_bytes_or_path, target_sr=TARGET_SR)
#     #     y = pad_or_trim(y, duration_s=3.0, sr=sr)
#     #     mel = logmel(y, sr)
#     #     x = torch.from_numpy(mel[None, None, :, :]).to(self.device)
#     #     x.requiresgrad(True)

#     #     cam_util = SpectrogramGradCAM(self.model, target_layer_name="features.6")

#     #     # Gradients ON for Grad-CAM
#     #     with torch.enable_grad():
#     #         cam, logits = cam_util(x, class_idx=None)  # cam: [1, H, W]

#     #     # turn logits -> probs for display
#     #     if isinstance(logits, torch.Tensor):
#     #         probs = F.softmax(logits, dim=-1).detach().cpu().numpy()[0]
#     #     else:
#     #         probs = F.softmax(torch.tensor(logits), dim=-1).cpu().numpy()[0]

#     #     return {"cam": cam[0].tolist(),
#     #             "probs": {"human": float(probs[0]), "ai": float(probs[1])}}
#     def explain(self, wav_bytes_or_path) -> dict:
#         # IMPORTANT: no @torch.inference_mode() here
#         self.model.eval()

#         # 1) build input
#         y, sr = load_audio(wav_bytes_or_path, target_sr=TARGET_SR)
#         y = pad_or_trim(y, duration_s=3.0, sr=sr)
#         mel = logmel(y, sr)                                 # (n_mels, T)
#         x = torch.from_numpy(mel[None, None, :, :]).to(self.device)
#         x.requiresgrad(True)                              # <— this was the typo

#         # 2) Grad-CAM
#         from .utils.gradcam import SpectrogramGradCAM
#         cam_util = SpectrogramGradCAM(self.model, target_layer_name="features.6")
#         with torch.enable_grad():                           # grads ON
#             cam, logits = cam_util(x, class_idx=None)       # cam: [1,H,W], logits: [1,2] (numpy)

#         # 3) probs for display
#         logits_t = torch.tensor(logits)
#         probs = F.softmax(logits_t, dim=-1).cpu().numpy()[0]
#         return {
#             "cam": cam[0].tolist(),
#             "probs": {"human": float(probs[0]), "ai": float(probs[1])}
#         }

#--------------------------------------------------------------------------------

import os
import numpy as np
import torch
import torch.nn.functional as F

from .models.cnn_melspec import TinyMelCNN
from .utils.audio import load_audio, pad_or_trim, logmel, heuristic_features, TARGET_SR
from .utils.gradcam import SpectrogramGradCAM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Env-tunable knobs (safe defaults)
USE_HEURISTIC = os.getenv("DETECTOR_ALLOW_HEURISTIC", "0") == "1"  # default OFF
AI_THRESHOLD  = float(os.getenv("DETECTOR_AI_THRESHOLD", "0.50"))  # decision threshold
AI_PROB_BIAS  = float(os.getenv("DETECTOR_AI_BIAS", "0.00"))       # e.g., 0.03 to lean AI slightly

class Detector:
    def __init__(self, weights_path: str | None = None, use_cuda: bool | None = None):
        self.device = DEVICE if use_cuda is None else ("cuda" if use_cuda else "cpu")
        self.model = TinyMelCNN().to(self.device)
        self.trained = False
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.trained = True

    @torch.inference_mode()
    def predict_proba(self, wav_bytes_or_path) -> dict:
        """Return dict: human/ai probs, label, threshold, trained flag."""
        y, sr = load_audio(wav_bytes_or_path, target_sr=TARGET_SR)
        y = pad_or_trim(y, duration_s=3.0, sr=sr)
        mel = logmel(y, sr)
        x = torch.from_numpy(mel[None, None]).to(self.device)

        logits = self.model(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]  # [human, ai]

        # Heuristic only if explicitly enabled AND no trained weights present
        if (not self.trained) and USE_HEURISTIC:
            feats = heuristic_features(y, sr)               # [zcr, centroid, flatness, ...]
            zcr, centroid, flatness = float(feats[0]), float(feats[1]), float(feats[2])
            score = 0.5 * flatness + 1e-6 * centroid + 0.25 * zcr
            score = float(np.clip(score, 0.0, 1.0))
            ai_prob = float(0.6 * probs[1] + 0.4 * score)   # lighter mix than before
            probs = np.array([1.0 - ai_prob, ai_prob], dtype=np.float32)

        # Optional bias: tiny positive values (e.g., +0.03) can reduce false-human on replayed TTS
        probs[1] = float(np.clip(probs[1] + AI_PROB_BIAS, 0.0, 1.0))
        probs[0] = float(1.0 - probs[1])

        label = "ai" if probs[1] >= AI_THRESHOLD else "human"
        return {
            "human": float(probs[0]),
            "ai": float(probs[1]),
            "label": label,
            "threshold": AI_THRESHOLD,
            "trained": self.trained,
        }

    def explain(self, wav_bytes_or_path) -> dict:
        """Grad-CAM explanation. Requires grads enabled."""
        self.model.eval()

        # build input
        y, sr = load_audio(wav_bytes_or_path, target_sr=TARGET_SR)
        y = pad_or_trim(y, duration_s=3.0, sr=sr)
        mel = logmel(y, sr)                                 # (n_mels, T)
        x = torch.from_numpy(mel[None, None, :, :]).to(self.device)
        x.requires_grad_(True)

        cam_util = SpectrogramGradCAM(self.model, target_layer_name="features.6")

        # Gradients ON for Grad-CAM
        with torch.enable_grad():
            cam, logits = cam_util(x, class_idx=None)       # cam: [1,H,W], logits: [1,2] (numpy)

        # probs for display
        logits_t = torch.tensor(logits)
        probs = F.softmax(logits_t, dim=-1).cpu().numpy()[0]

        return {
            "cam": cam[0].tolist(),
            "probs": {"human": float(probs[0]), "ai": float(probs[1])}
        }