import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import uuid
from datetime import datetime
from pathlib import Path

# Detect device (Mac with M1/M2/M3/M4)
device = "mps" if torch.backends.mps.is_available() else "cpu"
map_location = torch.device(device)

torch_load_original = torch.load


def patched_torch_load(*args, **kwargs):
    if "map_location" not in kwargs:
        kwargs["map_location"] = map_location
    return torch_load_original(*args, **kwargs)


torch.load = patched_torch_load

model = ChatterboxTTS.from_pretrained(device=device)
text = """

"""



# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "Joshua Graham teaches you how to be a man.wav"
# "Joshua Graham teaches you how to be a man.wav"
wav = model.generate(
    text=text,
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.5,
    cfg_weight=0.4,
)

ta.save(f"ttt-{datetime.now().isoformat()}.wav", wav, model.sr)
