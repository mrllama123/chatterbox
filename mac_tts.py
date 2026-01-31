import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import uuid
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import argparse

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


def generate_and_combine_audio(
    text_chunks: list[str], audio_prompt_path: str, output_filename: str
) -> None:
    """Generate audio for text chunks and combine them."""

    audio_segments = []

    print(f"Processing {len(text_chunks)} text chunks...")

    for i, chunk in enumerate(text_chunks):
        print(f"Generating audio for chunk {i+1}/{len(text_chunks)}")
        wav = model.generate(
            text=chunk,
            audio_prompt_path=audio_prompt_path,
            exaggeration=0.5,
            cfg_weight=0.4,
        )
        audio_segments.append(wav)

    # Combine all audio segments
    combined_audio = torch.cat(audio_segments, dim=-1)

    # Save the combined audio
    ta.save(output_filename, combined_audio, model.sr)
    print(f"Combined audio saved as: {output_filename}")


def read_text_xml_file(file_path: str) -> list[str]:
    tree = ET.parse(file_path)
    root = tree.getroot()
    chunks = sorted(root.find("chunks"), key=lambda chunk: int(chunk.attrib["id"]))
    return [chunk.text for chunk in chunks]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate and combine audio from text chunks."
    )
    parser.add_argument(
        "-x",
        "--xml_file",
        type=str,
        required=True,
        help="Path to the XML file containing text chunks.",
    )
    parser.add_argument(
        "-a",
        "--audio_prompt",
        type=str,
        default="Joshua Graham teaches you how to be a man.wav",
        help="Path to the audio prompt file.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=f"tts-{datetime.now().isoformat()}.wav",
        help="Output filename for the combined audio.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    text_chunks = read_text_xml_file(args.xml_file)
    generate_and_combine_audio(
        text_chunks,
        args.audio_prompt,
        args.output_file,
    )


if __name__ == "__main__":
    main()
