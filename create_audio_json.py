import whisperx
import gc 
import json

import os
import json
from transformers import pipeline
from moviepy.editor import *
from IPython.display import Audio
from pprint import pprint

# PATH
audio_path = 'audio_path'
video_path = 'videos'
audio_json_path = 'audio_json'

device = "cuda" 
batch_size = 32 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

model = whisperx.load_model("large-v2", device, compute_type=compute_type)
videos = os.listdir(video_path)

for vid in videos:
    video = VideoFileClip(os.path.join(video_path, vid))
    audio_file = os.path.join(audio_path, vid).replace('mp4', 'wav')
    video.audio.write_audiofile(audio_file)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    #Delete audio after processing
    if os.path.isfile(audio_file):
        os.remove(audio_file)

    json_file = os.path.join(audio_json_path, vid.replace('mp4', 'json'))
    # print(result["segments"])
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(result['segments'], f, ensure_ascii=False, indent=4)
        
