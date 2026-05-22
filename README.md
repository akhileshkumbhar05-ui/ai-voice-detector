# AI Voice Detector

AI Voice Detector is a hackathon-style machine learning project that detects whether a short speech clip is **human-recorded** or **AI-generated**. The system takes an uploaded or microphone-recorded audio clip, converts it into a log-mel spectrogram, runs it through a lightweight convolutional neural network, and returns:

- Probability that the clip is human speech
- Probability that the clip is AI-generated speech
- Final predicted label
- Grad-CAM spectrogram heatmap showing which time-frequency regions influenced the model

The project also includes scripts for generating AI speech clips with ElevenLabs, converting MP3 files to 16 kHz WAV format, training a PyTorch CNN classifier, and running an interactive Gradio web demo.

Try the deployed app here: [Voice Guard on Hugging Face Spaces](https://huggingface.co/spaces/varunkul/Voice-guard)

---

## Project Goal

The goal of this project is to build a lightweight, explainable AI voice detection system that can classify short audio clips as either:

- `human`
- `ai`

The project is designed as a practical prototype for detecting synthetic speech in real time or near real time. It focuses on a complete end-to-end workflow:

1. Collect or generate audio data
2. Normalize all clips to a consistent audio format
3. Convert audio into log-mel spectrogram features
4. Train a compact CNN classifier
5. Run inference through a web UI
6. Explain model decisions using Grad-CAM heatmaps

---

## Current Status

This repository contains a working prototype with:

- A trained PyTorch model checkpoint
- A Gradio-based web interface
- Audio preprocessing utilities
- A custom CNN model
- A training pipeline
- Grad-CAM explainability
- ElevenLabs-based AI speech generation scripts
- A small local dataset of AI and human speech samples
- Docker support for running the app

This is not a production-grade detector yet. It is a hackathon/prototype system trained on a limited dataset, so predictions should be treated as experimental.

---

## Main Features

### 1. Human vs AI Speech Classification

The detector accepts a short audio clip and returns class probabilities:

```json
{
  "human": 0.23,
  "ai": 0.77,
  "label": "ai",
  "threshold": 0.5,
  "trained": true
}
```

The model predicts `ai` when the AI probability is greater than or equal to the configured threshold.

---

### 2. Gradio Web Interface

The app provides a simple browser-based UI where users can:

- Record audio from a microphone
- Upload an audio file
- Click **Analyze**
- View model probabilities
- View an explanation heatmap
- Optionally run a provenance check

The active app entry point is:

```bash
python app/app.py
```

The Gradio interface launches from `app/app.py`.

---

### 3. Explainable Heatmap with Grad-CAM

The model includes a Grad-CAM utility for spectrogram-based explanations.

The Grad-CAM output highlights the regions of the mel spectrogram that influenced the model decision most strongly.

This helps make the system more interpretable than a plain binary classifier.

The explanation logic is implemented in:

```text
app/utils/gradcam.py
```

The detector calls Grad-CAM from:

```text
app/inference.py
```

The Gradio app displays the heatmap as an image using a `magma` colormap from Matplotlib.

---

### 4. ElevenLabs AI Speech Generation

The repository includes a generation pipeline for creating AI voice clips using ElevenLabs text-to-speech.

The generation code is in:

```text
gen_clips.py
app/elevenlabs_tools.py
```

The generation script uses 10 different ElevenLabs voices:

- Adam
- Alice
- Aria
- Brian
- Bill
- Charlotte
- Clyde
- Drew
- Freya
- Gigi

The dataset generation script contains 200 scripted sentences total, with 20 sentences per voice. The prompts cover varied speaking styles such as:

- News anchor
- Friendly conversation
- British formal speech
- Energetic young speaker
- Calm meditation
- Elderly storyteller
- Technical presenter
- Audiobook narration
- Sports commentator
- Curious childlike questions

Generated MP3 files are saved to:

```text
data/raw/ai_mp3/
```

They are then converted to 16 kHz mono WAV files and saved to:

```text
data/raw/ai/
```

---

### 5. Audio Preprocessing

All audio is normalized into a consistent format before training or inference.

The preprocessing utility is implemented in:

```text
app/utils/audio.py
```

The audio pipeline does the following:

1. Loads audio from a file path or raw bytes
2. Converts audio to mono
3. Resamples to 16 kHz
4. Normalizes amplitude
5. Pads or trims the clip to a fixed duration
6. Converts the waveform into a log-mel spectrogram

Important constants and settings:

```python
TARGET_SR = 16000
clip duration = 3.0 seconds
n_mels = 64
n_fft = 1024
hop_length = 256
fmin = 20
fmax = sr // 2
```

The classifier is trained on log-mel spectrograms rather than raw waveforms.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── .env
├── convert.py
├── gen_clips.py
├── app/
│   ├── __init__.py
│   ├── app.py
│   ├── inference.py
│   ├── train.py
│   ├── elevenlabs_tools.py
│   ├── models/
│   │   ├── cnn_melspec.py
│   │   └── weights/
│   │       ├── cnn_melspec.pth
│   │       └── cnn_melspec.last.pth
│   └── utils/
│       ├── audio.py
│       ├── convert_mp3_to_wav.py
│       └── gradcam.py
├── data/
│   └── raw/
│       ├── ai/
│       ├── ai_mp3/
│       └── human/
├── human/
│   └── original AAC human recordings
├── docker/
│   └── Dockerfile
└── notebooks/
    └── 01_error_analysis.ipynb
```

---

## Important Files

### `app/app.py`

Main Gradio application.

Responsibilities:

- Loads environment variables
- Loads the trained model checkpoint
- Accepts microphone or uploaded audio from Gradio
- Converts Gradio audio input into WAV bytes
- Runs prediction using `Detector`
- Runs Grad-CAM explanation
- Displays prediction probabilities and heatmap
- Provides an optional provenance check button

Default model path:

```text
app/models/weights/cnn_melspec.pth
```

This can be overridden with:

```text
MODEL_WEIGHTS_PATH
```

---

### `app/inference.py`

Inference wrapper around the trained CNN.

Main class:

```python
Detector
```

Responsibilities:

- Loads `TinyMelCNN`
- Loads model weights if available
- Converts audio into log-mel spectrogram features
- Runs model inference
- Applies softmax to get probabilities
- Applies a configurable AI threshold
- Optionally applies a fallback heuristic if no trained weights are present
- Runs Grad-CAM explanations

Environment-configurable inference settings:

```text
MODEL_WEIGHTS_PATH
DETECTOR_ALLOW_HEURISTIC
DETECTOR_AI_THRESHOLD
DETECTOR_AI_BIAS
```

Default values:

```text
DETECTOR_ALLOW_HEURISTIC=0
DETECTOR_AI_THRESHOLD=0.50
DETECTOR_AI_BIAS=0.00
```

The detector returns:

```json
{
  "human": 0.0,
  "ai": 1.0,
  "label": "ai",
  "threshold": 0.5,
  "trained": true
}
```

---

### `app/models/cnn_melspec.py`

Defines the CNN model used for classification.

Main model:

```python
TinyMelCNN
```

Architecture:

```text
Input: 1 x n_mels x time

Conv2d: 1 -> 16
BatchNorm2d
ReLU
MaxPool2d

Conv2d: 16 -> 32
BatchNorm2d
ReLU
MaxPool2d

Conv2d: 32 -> 64
BatchNorm2d
ReLU
AdaptiveAvgPool2d(8 x 8)

Flatten
Linear: 64*8*8 -> 128
ReLU
Dropout(0.2)
Linear: 128 -> 2
```

Output classes:

```text
0 = human
1 = ai
```

The model checkpoint contains approximately 548k parameters.

---

### `app/train.py`

Training pipeline for the detector.

Main responsibilities:

- Loads data from a folder with `human/` and `ai/` subfolders
- Splits data into train and validation sets
- Applies class-specific audio augmentation
- Converts audio into log-mel spectrograms
- Trains `TinyMelCNN`
- Uses class-weighted cross entropy to handle imbalance
- Supports GPU training when CUDA is available
- Supports automatic mixed precision
- Saves best and latest model checkpoints

Expected training data format:

```text
data/raw/
├── human/
│   └── *.wav
└── ai/
    └── *.wav
```

Default training command:

```bash
python -m app.train --data_dir data/raw --out app/models/weights/cnn_melspec.pth
```

Useful options:

```bash
python -m app.train \
  --data_dir data/raw \
  --out app/models/weights/cnn_melspec.pth \
  --epochs 10 \
  --batch_size 32 \
  --grad_accum 2 \
  --lr 1e-3 \
  --val_ratio 0.15 \
  --clip_seconds 3.0
```

The script saves:

```text
app/models/weights/cnn_melspec.pth
app/models/weights/cnn_melspec.last.pth
```

---

### `app/utils/audio.py`

Audio loading and feature extraction utilities.

Main functions:

```python
load_audio()
pad_or_trim()
logmel()
heuristic_features()
```

The `heuristic_features()` function extracts lightweight audio features such as:

- Zero crossing rate
- Spectral centroid
- Spectral flatness
- Spectral rolloff
- RMS energy
- MFCCs

These heuristic features are only used if heuristic fallback is explicitly enabled.

---

### `app/utils/gradcam.py`

Implements Grad-CAM for the spectrogram CNN.

Main class:

```python
SpectrogramGradCAM
```

The target layer used by default is:

```text
features.6
```

This corresponds to the third convolutional layer in `TinyMelCNN`.

The Grad-CAM process:

1. Runs a forward pass
2. Selects the predicted class score
3. Backpropagates from that score
4. Averages gradients spatially
5. Weights feature maps by those gradients
6. Applies ReLU
7. Normalizes the heatmap to `[0, 1]`

The output is returned as a NumPy array and displayed in the Gradio app.

---

### `app/elevenlabs_tools.py`

Contains ElevenLabs helper functions.

Main functions:

```python
generate_tts_dataset()
check_ai_speech()
```

`generate_tts_dataset()` calls the ElevenLabs text-to-speech API and saves MP3 files.

The default ElevenLabs model ID is:

```text
eleven_monolingual_v1
```

`check_ai_speech()` is currently a stub. It returns:

```json
{
  "supported": false,
  "prob_ai": null,
  "provider": "elevenlabs",
  "note": "Classifier not enabled in this template."
}
```

So the provenance button exists in the UI, but the current implementation does not perform a real external provenance check.

---

### `app/utils/convert_mp3_to_wav.py`

Utility script for converting MP3 files to 16 kHz mono WAV files.

Example:

```bash
python -m app.utils.convert_mp3_to_wav --src data/raw/ai_mp3 --dst data/raw/ai
```

The conversion uses:

- `librosa` for loading and resampling
- `soundfile` for writing WAV files

---

### `gen_clips.py`

AI speech dataset generation script.

Responsibilities:

- Loads `ELEVEN_API_KEY` from `.env`
- Uses 10 ElevenLabs voices
- Generates 200 AI speech clips
- Saves MP3 files to `data/raw/ai_mp3`
- Converts MP3 files to 16 kHz mono WAV files in `data/raw/ai`

Run:

```bash
python gen_clips.py
```

Required environment variable:

```text
ELEVEN_API_KEY
```

---

### `convert.py`

Utility for flattening and renaming files.

It recursively walks through a source folder, copies files into one output folder, and renames them using the pattern:

```text
elab_0001.ext
elab_0002.ext
...
```

Current hardcoded paths:

```python
root_folder = "data/raw/ai_mp3"
output_folder = "data/raw/ai"
```

Use this carefully because it copies files as-is and does not perform audio format conversion.

For MP3-to-WAV conversion, prefer:

```text
app/utils/convert_mp3_to_wav.py
```

---

### `docker/Dockerfile`

Docker configuration for running the Gradio app.

Current Dockerfile:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY .env ./
EXPOSE 7860
CMD ["python", "app/app.py"]
```

Build:

```bash
docker build -f docker/Dockerfile -t ai-voice-detector .
```

Run:

```bash
docker run -p 7860:7860 ai-voice-detector
```

Security note: the current Dockerfile copies `.env` into the image. That is acceptable only for local experimentation. For public repositories or shared images, do not copy `.env` into the image. Use runtime environment variables or `--env-file` instead.

---

### `notebooks/01_error_analysis.ipynb`

Placeholder notebook for future error analysis. In the current ZIP, this notebook does not contain analysis cells yet.

---

## Dataset Included in This Repository

The ZIP contains local audio data.

### AI Speech Data

```text
data/raw/ai/
```

Contains:

```text
200 WAV files
```

Format:

```text
16 kHz mono WAV
```

These are generated AI speech clips.

---

### AI MP3 Source Data

```text
data/raw/ai_mp3/
```

Contains:

```text
200 MP3 files
```

These are the original ElevenLabs-generated MP3 files before conversion to WAV.

---

### Human Speech Data

```text
data/raw/human/
```

Contains:

```text
50 WAV files
```

Format:

```text
16 kHz mono WAV
```

These are human speech samples used for training and validation.

---

### Original Human Recordings

```text
human/
```

Contains:

```text
50 AAC files
```

These appear to be original human recordings before conversion into the training-ready WAV format.

---

## Dataset Balance

The current dataset is imbalanced:

```text
AI clips:    200
Human clips: 50
```

The training code compensates for this imbalance using class-weighted cross entropy.

Still, because the dataset is small and skewed, the model should be treated as a prototype rather than a robust real-world detector.

---

## Technologies Used

### Machine Learning and Deep Learning

- Python
- PyTorch
- TorchAudio
- NumPy
- SciPy
- librosa
- soundfile
- audiomentations

### Audio Processing

- 16 kHz mono audio normalization
- Log-mel spectrogram extraction
- MP3-to-WAV conversion
- Audio augmentation
- Spectral feature extraction
- MFCC-based heuristic features

### Model Explainability

- Grad-CAM
- Spectrogram heatmap visualization
- Matplotlib colormaps

### Web App

- Gradio

### API and Environment Utilities

- requests
- python-dotenv
- pydantic
- FastAPI
- Uvicorn

Note: FastAPI and Uvicorn are listed in `requirements.txt`, but the current active app is Gradio-based. There is no active FastAPI server implementation in the current source files.

### Data Generation

- ElevenLabs text-to-speech API

### Containerization

- Docker
- Python 3.11 slim image

### Development Tools

- Black formatter
- Jupyter Notebook placeholder

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-voice-detector.git
cd ai-voice-detector
```

---

### 2. Create a Virtual Environment

On Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a local `.env` file if you want to use ElevenLabs generation.

Example:

```text
ELEVEN_API_KEY=your_elevenlabs_api_key_here
ELEVEN_VOICE_ID=your_default_voice_id_here
MODEL_WEIGHTS_PATH=app/models/weights/cnn_melspec.pth
DETECTOR_AI_THRESHOLD=0.50
DETECTOR_AI_BIAS=0.00
DETECTOR_ALLOW_HEURISTIC=0
```

Do not commit `.env` to GitHub.

Recommended `.gitignore` entries:

```gitignore
.env
*.env
__pycache__/
*.py[cod]
.venv/
venv/
.ipynb_checkpoints/
.DS_Store
Thumbs.db
```

---

## Running the App

From the repository root:

```bash
python app/app.py
```

Or:

```bash
python -m app.app
```

Then open the local Gradio URL shown in the terminal.

The app allows you to:

1. Upload or record audio
2. Click **Analyze**
3. View probabilities and predicted label
4. View Grad-CAM heatmap
5. Optionally click **Provenance Check**

---

## Training the Model

To train the CNN from scratch:

```bash
python -m app.train --data_dir data/raw --out app/models/weights/cnn_melspec.pth
```

More complete example:

```bash
python -m app.train \
  --data_dir data/raw \
  --out app/models/weights/cnn_melspec.pth \
  --epochs 10 \
  --batch_size 32 \
  --grad_accum 2 \
  --lr 1e-3 \
  --val_ratio 0.15 \
  --clip_seconds 3.0 \
  --seed 42
```

Use CPU explicitly:

```bash
python -m app.train --data_dir data/raw --cpu
```

Use zero workers on Windows if multiprocessing causes issues:

```bash
python -m app.train --data_dir data/raw --workers 0
```

---

## Training Pipeline Details

The training process:

1. Reads files from `data/raw/human` and `data/raw/ai`
2. Assigns labels:
   - `0 = human`
   - `1 = ai`
3. Randomly shuffles the dataset
4. Splits into training and validation sets
5. Pads or trims each clip to 3 seconds
6. Applies augmentation during training
7. Converts audio to log-mel spectrogram
8. Trains the CNN using cross entropy loss
9. Saves the latest checkpoint every epoch
10. Saves the best checkpoint when validation accuracy improves

---

## Audio Augmentation

Human clips receive mild natural perturbations:

- Gaussian noise
- Small time stretch
- Small pitch shift
- Gain adjustment

AI clips receive replay-aware augmentation:

- Band-pass filtering
- Gaussian noise
- Time stretch
- Gain adjustment
- Optional clipping distortion
- Optional MP3 compression when supported

This is intended to make the detector more robust to microphone, speaker, compression, and replay artifacts.

---

## Generating AI Voice Data

To generate AI speech clips with ElevenLabs:

1. Add your ElevenLabs API key to `.env`
2. Run:

```bash
python gen_clips.py
```

The script will:

1. Generate MP3 clips using ElevenLabs
2. Save them under `data/raw/ai_mp3`
3. Convert them to 16 kHz mono WAV files under `data/raw/ai`

---

## Converting MP3 Files to WAV

Use:

```bash
python -m app.utils.convert_mp3_to_wav --src data/raw/ai_mp3 --dst data/raw/ai
```

This converts all MP3 files in the source folder to 16 kHz mono WAV files.

---

## Model Output

The model returns two probabilities:

```text
human
ai
```

It also returns a final label based on the AI threshold.

Example:

```json
{
  "human": 0.184,
  "ai": 0.816,
  "label": "ai",
  "threshold": 0.5,
  "trained": true
}
```

---

## Explainability Output

The explanation heatmap is generated from the model's convolutional feature maps.

The heatmap shows which spectrogram regions had the greatest influence on the predicted class.

Bright/high-activation regions indicate time-frequency areas that contributed more strongly to the model's decision.

---

## Known Limitations

This project is a prototype and has several important limitations:

1. The dataset is small.
2. The dataset is imbalanced, with more AI clips than human clips.
3. AI clips are generated from ElevenLabs, so the model may overfit to ElevenLabs-style speech.
4. The model may not generalize well to other TTS providers.
5. Real-world audio conditions are more varied than the current dataset.
6. The provenance check button currently calls a stub, not a real external detector.
7. The notebook for error analysis is currently empty.
8. The current Dockerfile copies `.env`, which should be changed before public deployment.
9. No formal benchmark metrics are included in the repository.
10. The model should not be used for high-stakes decisions without more data, calibration, testing, and validation.

---

## Recommended Future Improvements

Useful next steps:

1. Add more human voices across genders, accents, microphones, rooms, and recording devices.
2. Add AI speech from multiple providers, not only ElevenLabs.
3. Add train/validation/test split with a held-out speaker/provider split.
4. Track metrics such as accuracy, precision, recall, F1, ROC-AUC, and equal error rate.
5. Add a confusion matrix and error analysis notebook.
6. Calibrate model probabilities.
7. Add support for longer clips using sliding windows.
8. Add a FastAPI inference endpoint.
9. Replace `.env` copying in Docker with runtime environment variables.
10. Add model versioning and dataset versioning.
11. Use Git LFS for model weights and large audio files if the repository grows.
12. Add CI checks for formatting, linting, and basic inference tests.

---

## Security Notes

Do not commit secrets to GitHub.

The following file should stay local:

```text
.env
```

The `.env` file may contain:

```text
ELEVEN_API_KEY
ELEVEN_VOICE_ID
```

If these credentials were accidentally pushed to GitHub, revoke and regenerate the API key.

---

## Suggested GitHub Description

```text
Explainable AI voice detector that classifies short speech clips as human or AI-generated using log-mel spectrograms, a PyTorch CNN, Grad-CAM heatmaps, and a Gradio demo.
```

---

## Suggested Topics

```text
ai-voice-detection
deepfake-audio
synthetic-speech
pytorch
gradio
grad-cam
audio-classification
mel-spectrogram
elevenlabs
machine-learning
```

---

## License

Add the license used by this repository here.

If this repository was initialized with an MIT License on GitHub, keep the existing `LICENSE` file and use:

```text
MIT License
```
