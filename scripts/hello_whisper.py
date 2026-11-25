import librosa
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")
audio, sr = librosa.load("test.wav", sr=16000) # Whisper needs 16khz

result = asr({"raw": audio, "sampling_rate": sr}, generate_kwargs = {"language": "Icelandic", "task": "transcribe"})

print(result["text"])


