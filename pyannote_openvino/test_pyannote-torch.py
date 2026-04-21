from pyannote.audio import Pipeline
import time
import argparse
import torch

# usage: python test_pyannote-torch.py "..\samples\audio.wav" --device "xpu"
# python 3.14.4
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
# pip install pyannote.audio
# conda install ffmpeg
parser = argparse.ArgumentParser(description="Benchmark Torch Pyannote")
parser.add_argument("audio_file", help="The name of the audio file to diarize.")
parser.add_argument("--device", default="xpu", help="Intel xpu,cpu")
args = parser.parse_args()

device = args.device
audio_file = args.audio_file

# Dynamically select the best available device
if "xpu" in device and not torch.xpu.is_available():
    print("FAILURE: xpu is not available.")
    quit()

if "xpu" in device:
    device = torch.device("xpu")  # Intel GPU
else:
    device = torch.device("cpu") 

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")

pipeline = pipeline.to(device)

# run the pipeline locally on your computer
start_t = time.time()
output = pipeline(audio_file)
end_t = time.time()

# print the predicted speaker diarization
for turn, speaker in output.speaker_diarization:
    print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")


print(f"{device} Diarization took: {(end_t-start_t)} seconds")