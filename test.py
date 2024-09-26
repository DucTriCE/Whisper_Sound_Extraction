import torch
torch.set_num_threads(1)

import os
import json
from transformers import pipeline
from moviepy.editor import *
from IPython.display import Audio
from pprint import pprint

from silero_vad import (load_silero_vad, read_audio, VADIterator)

SAMPLING_RATE = 16000
audio_path = 'audio_path'
video_path = 'videos'
audio_json_path = 'audio_json'

whisper = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", device='cuda:0')
model = load_silero_vad(onnx=False)
videos = os.listdir(video_path)

for vid in videos:
    video = VideoFileClip(os.path.join(video_path, vid))
    audio_file = os.path.join(audio_path, vid).replace('mp4', 'wav')
    video.audio.write_audiofile(audio_file)

    vad_iterator = VADIterator(model, sampling_rate=SAMPLING_RATE)
    wav = read_audio(audio_file, sampling_rate=SAMPLING_RATE)

    window_size_samples = 512
    speech_segments = []
    current_start = None

    for i in range(0, len(wav), window_size_samples):
        audio_chunk = wav[i: i + window_size_samples]
        print(audio_chunk)
        speech_dict = vad_iterator(audio_chunk, return_seconds=True)
        if speech_dict:
            if 'start' in speech_dict and current_start is None:
                current_start = speech_dict['start']
            elif 'end' in speech_dict and current_start is not None:
                speech_segments.append({'start': current_start, 'end': speech_dict['end']})
                current_start = None

    vad_iterator.reset_states()

    all_transcriptions = []
    for segment in speech_segments:
        start_time = segment['start']
        end_time = segment['end']
        transcription = whisper(audio_file, return_timestamps=True, task="transcribe", start=start_time, end=end_time)
        all_transcriptions.append(transcription)

    json_file = os.path.join(audio_json_path, vid.replace('mp4', 'json'))
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_transcriptions, f, ensure_ascii=False, indent=4)

    print(f"Transcription for {vid} saved to {json_file}")
    break
