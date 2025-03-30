import os
import glob
import argparse
import numpy as np
import torch
import pandas as pd
import json
import hashlib
import pygame
import time
from msclap import CLAP
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

class SoundMatcher:
    def __init__(self, audio_dir, use_cuda=False, cache_dir="cache", force_recompute=False):
        """
        Initialize the SoundMatcher with a directory of audio files
        
        Args:
            audio_dir (str): Path to directory containing audio files
            use_cuda (bool): Whether to use CUDA for CLAP model
            cache_dir (str): Directory to store cached embeddings
            force_recompute (bool): Force recomputation of embeddings even if cache exists
        """
        self.audio_dir = audio_dir
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.cache_dir = cache_dir
        self.force_recompute = force_recompute
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load CLAP model
        print("Loading CLAP model...")
        self.clap_model = CLAP(version='2023', use_cuda=use_cuda)
        
        # Find all audio files
        self.audio_files = self._get_audio_files()
        print(f"Found {len(self.audio_files)} audio files")
        
        # Pre-compute audio embeddings
        self.audio_embeddings = None
        self.index_to_file = {}
        self._load_or_compute_audio_embeddings()
    
    def _get_audio_files(self):
        """Get all audio files in the directory"""
        audio_extensions = ['.wav', '.mp3', '.ogg', '.flac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(self.audio_dir, f"**/*{ext}"), recursive=True))
        
        return sorted(audio_files)  # Sort to ensure consistent ordering
    
    def _get_cache_filename(self):
        """Generate a consistent filename for the cache based on the audio files"""
        # Create a hash of the audio directory path
        hasher = hashlib.md5()
        hasher.update(self.audio_dir.encode('utf-8'))
        
        # Add the number of files to the hash
        hasher.update(str(len(self.audio_files)).encode('utf-8'))
        
        # Generate a hash digest
        hash_digest = hasher.hexdigest()
        
        return hash_digest
    
    def _get_cache_path(self):
        """Get the path to the cache file"""
        cache_filename = self._get_cache_filename()
        return os.path.join(self.cache_dir, f"audio_embeddings_{cache_filename}.npz")
    
    def _get_metadata_path(self):
        """Get the path to the metadata file"""
        cache_filename = self._get_cache_filename()
        return os.path.join(self.cache_dir, f"metadata_{cache_filename}.json")
    
    def _save_embeddings(self):
        """Save embeddings to cache file"""
        cache_path = self._get_cache_path()
        metadata_path = self._get_metadata_path()
        
        # Convert tensor to numpy for saving
        embeddings_np = self.audio_embeddings.detach().cpu().numpy()
        np.savez_compressed(cache_path, embeddings=embeddings_np)
        
        # Save mapping from index to file
        with open(metadata_path, 'w') as f:
            json.dump(self.index_to_file, f)
        
        print(f"Saved embeddings to {cache_path}")
    
    def _load_embeddings(self):
        """Load embeddings from cache file"""
        cache_path = self._get_cache_path()
        metadata_path = self._get_metadata_path()
        
        if os.path.exists(cache_path) and os.path.exists(metadata_path):
            print(f"Loading cached embeddings from {cache_path}")
            
            # Load embeddings
            data = np.load(cache_path)
            embeddings_np = data['embeddings']
            self.audio_embeddings = torch.tensor(embeddings_np, dtype=torch.float32)
            
            # Load mapping from index to file
            with open(metadata_path, 'r') as f:
                # JSON keys are strings, convert them back to integers
                index_to_file_str = json.load(f)
                self.index_to_file = {int(k): v for k, v in index_to_file_str.items()}
            
            return True
        
        return False
    
    def _load_or_compute_audio_embeddings(self):
        """Load embeddings from cache if available, otherwise compute them"""
        if not self.force_recompute and self._load_embeddings():
            print(f"Loaded embeddings with shape: {self.audio_embeddings.shape}")
            return
        
        print("Computing audio embeddings...")
        
        # Process in batches to avoid memory issues
        batch_size = 50
        all_embeddings = []
        
        for i in tqdm(range(0, len(self.audio_files), batch_size)):
            batch_files = self.audio_files[i:i+batch_size]
            batch_embeddings = self.clap_model.get_audio_embeddings(batch_files)
            
            # Convert to PyTorch tensor if it's not already
            if not isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = torch.tensor(batch_embeddings, dtype=torch.float32)
            else:
                # Detach if it has gradients
                batch_embeddings = batch_embeddings.detach()
                
            all_embeddings.append(batch_embeddings)
            
            # Create mapping from index to file
            for j, file_path in enumerate(batch_files):
                self.index_to_file[i + j] = file_path
        
        # Stack all embeddings into a single tensor
        self.audio_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"Computed embeddings with shape: {self.audio_embeddings.shape}")
        
        # Save embeddings to cache
        self._save_embeddings()
    
    def find_matching_sounds(self, text_query, top_k=5):
        """
        Find the top-k sounds matching the text query
        
        Args:
            text_query (str): Text description of the sound
            top_k (int): Number of top matches to return
            
        Returns:
            list: List of (file_path, similarity_score) tuples
        """
        # Get text embedding
        text_embedding = self.clap_model.get_text_embeddings([text_query])
        
        # Convert to PyTorch tensor if it's not already
        if not isinstance(text_embedding, torch.Tensor):
            text_embedding = torch.tensor(text_embedding, dtype=torch.float32)
        else:
            # Detach if it has gradients
            text_embedding = text_embedding.detach()
        
        # Compute similarities
        similarities = self.clap_model.compute_similarity(self.audio_embeddings, text_embedding)
        
        # Convert to numpy for easier processing if it's a tensor
        if isinstance(similarities, torch.Tensor):
            similarities = similarities.detach().cpu().numpy()
        
        # Get top-k matches
        top_indices = np.argsort(similarities.flatten())[-top_k:][::-1]
        
        # Return file paths and scores
        results = []
        for idx in top_indices:
            file_path = self.index_to_file[idx]
            score = similarities[idx][0]
            results.append((file_path, score))
        
        return results
    
    def play_audio(self, file_path):
        """
        Play the audio file using pygame
        
        Args:
            file_path (str): Path to audio file
        """
        print(f"Playing: {os.path.basename(file_path)}")
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # Wait for the audio to finish playing
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)  # Wait 100 milliseconds between checks
            
            print("Playback complete")
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def visualize_audio(self, file_path):
        """
        Visualize the audio waveform and spectrogram
        
        Args:
            file_path (str): Path to audio file
        """
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Plot waveform and spectrogram
        plt.figure(figsize=(12, 8))
        
        # Waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title('Waveform')
        
        # Spectrogram
        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Match text queries to sounds using CLAP')
    parser.add_argument('--audio_dir', type=str, default='data/ESC-50-master/audio',
                        help='Directory containing audio files')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA for CLAP model')
    parser.add_argument('--query', type=str, 
                        help='Text query to match against sounds')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top matches to return')
    parser.add_argument('--cache_dir', type=str, default='cache',
                        help='Directory to store cached embeddings')
    parser.add_argument('--force_recompute', action='store_true',
                        help='Force recomputation of embeddings even if cache exists')
    parser.add_argument('--play', action='store_true',
                        help='Play the top matches')
    
    args = parser.parse_args()
    
    # Initialize sound matcher
    matcher = SoundMatcher(args.audio_dir, use_cuda=args.use_cuda, cache_dir=args.cache_dir, force_recompute=args.force_recompute)
    
    if args.query:
        # Find matching sounds
        matches = matcher.find_matching_sounds(args.query, top_k=args.top_k)
        
        # Print results
        print(f"\nTop {args.top_k} matches for query: '{args.query}'")
        print("-" * 50)
        for i, (file_path, score) in enumerate(matches):
            file_name = os.path.basename(file_path)
            print(f"{i+1}. {file_name} (score: {score:.4f})")
        
        # Play the top matches if requested
        if args.play and matches:
            print("\nPlaying top matches...")
            for i, (file_path, score) in enumerate(matches[:min(3, len(matches))]):
                print(f"\nPlaying match #{i+1}: {os.path.basename(file_path)}")
                matcher.play_audio(file_path)
                
                # Wait a bit between audio files
                time.sleep(0.5)
            
        # Visualize top match
        if matches:
            top_match = matches[0][0]
            print(f"\nVisualizing top match: {os.path.basename(top_match)}")
            matcher.visualize_audio(top_match)
    else:
        # Interactive mode
        while True:
            query = input("\nEnter a text query (or 'q' to quit): ")
            if query.lower() == 'q':
                break
                
            matches = matcher.find_matching_sounds(query, top_k=args.top_k)
            
            # Print results
            print(f"\nTop {args.top_k} matches for query: '{query}'")
            print("-" * 50)
            for i, (file_path, score) in enumerate(matches):
                file_name = os.path.basename(file_path)
                print(f"{i+1}. {file_name} (score: {score:.4f})")
            
            # Ask if user wants to play the top matches
            if matches:
                choice = input("\nPlay top 3 matches? (y/n): ")
                if choice.lower() == 'y':
                    for i, (file_path, score) in enumerate(matches[:min(3, len(matches))]):
                        print(f"\nPlaying match #{i+1}: {os.path.basename(file_path)}")
                        matcher.play_audio(file_path)
                        
                        # Wait a bit between audio files
                        time.sleep(0.5)
            
            # Ask if user wants to visualize top match
            if matches:
                choice = input("\nVisualize top match? (y/n): ")
                if choice.lower() == 'y':
                    top_match = matches[0][0]
                    print(f"Visualizing: {os.path.basename(top_match)}")
                    matcher.visualize_audio(top_match)

if __name__ == "__main__":
    main()
