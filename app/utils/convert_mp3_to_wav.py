
import os, glob
import soundfile as sf
import librosa

def mp3_to_wav16k(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    for mp3 in glob.glob(os.path.join(src_dir, "*.mp3")):
        y, sr = librosa.load(mp3, sr=16000, mono=True)
        base = os.path.splitext(os.path.basename(mp3))[0] + ".wav"
        out = os.path.join(dst_dir, base)
        sf.write(out, y, 16000, format="WAV")
        print("wrote", out)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder with mp3s")
    p.add_argument("--dst", required=True, help="output folder for wavs")
    args = p.parse_args()
    mp3_to_wav16k(args.src, args.dst)
