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
Hello wastelanders, Today i will discuss something that sadly is all to common in the modern world but overlooked, Addiction to the internet.

It has become such a common thing that we don't even acknowledge it and even when we are socializing half of the time we are glued to some screen looking 
at some stupid tiktok clip or some childish drama between content creators. This in the modern world is just normal. More and more we glue ourselves to the screen
just like how we did with tv's in the 20th century but i would argue its worse because of how accessible the internet is especially with mobile phones

You can get your fix on the go.

It's funny how we have all these dystopian movies and novels that talk about this very phenomenon. People becoming addicted and glued to screens while the world is
taken over by evil corporations and our freedom is being destroyed in front of our eyes  
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

ta.save(f"porn-addiction-{datetime.now().isoformat()}.wav", wav, model.sr)
