# CLAP Sound Matcher

This project uses CLAP (Contrastive Language-Audio Pretraining) to match text queries to sounds. It allows you to describe a sound in natural language, and the system will find the most similar sounds from a database.

## Features

- Match text descriptions to audio files using CLAP embeddings
- Interactive GUI for querying and visualizing results
- Audio playback of matched sounds
- Visualization of audio waveforms and spectrograms

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download the ESC-50 dataset:

```bash
python download_dataset.py
```

This will download and extract the ESC-50 dataset, which contains 2000 environmental sound recordings organized into 50 semantic classes.

## Usage

### Command Line Interface

```bash
python sound_matcher.py --query "dog barking" --top_k 5
```

Arguments:
- `--audio_dir`: Directory containing audio files (default: 'data/ESC-50-master/audio')
- `--use_cuda`: Use CUDA for CLAP model (flag)
- `--query`: Text query to match against sounds
- `--top_k`: Number of top matches to return (default: 5)

If you don't provide a query, the script will run in interactive mode.

### GUI Demo

```bash
python demo.py
```

Arguments:
- `--audio_dir`: Directory containing audio files (default: 'data/ESC-50-master/audio')
- `--use_cuda`: Use CUDA for CLAP model (flag)

## How It Works

1. The system loads the CLAP model, which has been trained to understand the relationship between text descriptions and audio content.
2. It computes embeddings for all audio files in the specified directory.
3. When you enter a text query, the system computes the text embedding and finds the audio files with the most similar embeddings.
4. The top matches are displayed, and you can visualize and play the sounds.

## Example Queries

Try these example queries:
- "dog barking loudly"
- "rain falling on a roof"
- "children playing"
- "car engine starting"
- "birds chirping in the forest"
- "door creaking"
- "footsteps on gravel"

## Notes

- The first run will download the CLAP model weights (~1GB) automatically.
- Computing audio embeddings may take some time, especially for large audio collections.
- For best results, be descriptive in your text queries.
