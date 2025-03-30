import os
import argparse
from sound_matcher import SoundMatcher
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import pygame
import time

class SoundMatcherApp:
    def __init__(self, root, audio_dir, use_cuda=False):
        self.root = root
        self.root.title("CLAP Sound Matcher")
        self.root.geometry("800x600")
        self.audio_dir = audio_dir
        self.use_cuda = use_cuda
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Create UI elements
        self.create_widgets()
        
        # Initialize sound matcher in a separate thread
        self.status_var.set("Loading CLAP model and computing audio embeddings...")
        self.matcher = None
        threading.Thread(target=self.init_matcher, daemon=True).start()
    
    def init_matcher(self):
        """Initialize the sound matcher in a background thread"""
        try:
            self.matcher = SoundMatcher(self.audio_dir, use_cuda=self.use_cuda)
            self.root.after(0, lambda: self.status_var.set("Ready! Enter a text query to find matching sounds."))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error initializing matcher: {str(e)}"))
    
    def create_widgets(self):
        """Create the UI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing...")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Query frame
        query_frame = ttk.LabelFrame(main_frame, text="Text Query", padding="10")
        query_frame.pack(fill=tk.X, pady=5)
        
        # Query entry
        self.query_var = tk.StringVar()
        query_entry = ttk.Entry(query_frame, textvariable=self.query_var, width=50)
        query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        query_entry.bind("<Return>", self.search_sounds)
        
        # Search button
        search_button = ttk.Button(query_frame, text="Search", command=self.search_sounds)
        search_button.pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD, height=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.results_text.config(state=tk.DISABLED)
        
        # Visualization frame
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create matplotlib figure
        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Play button
        self.play_button = ttk.Button(control_frame, text="Play Sound", command=self.play_sound, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        # Stop button
        self.stop_button = ttk.Button(control_frame, text="Stop Sound", command=self.stop_sound, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Current file label
        self.current_file_var = tk.StringVar()
        self.current_file_var.set("No file selected")
        current_file_label = ttk.Label(control_frame, textvariable=self.current_file_var)
        current_file_label.pack(side=tk.LEFT, padx=5)
        
        # Set focus to query entry
        query_entry.focus_set()
    
    def search_sounds(self, event=None):
        """Search for sounds matching the query"""
        if not self.matcher:
            self.status_var.set("Matcher not initialized yet. Please wait...")
            return
        
        query = self.query_var.get().strip()
        if not query:
            self.status_var.set("Please enter a text query")
            return
        
        self.status_var.set(f"Searching for: '{query}'")
        
        # Clear previous results
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Find matching sounds
        matches = self.matcher.find_matching_sounds(query, top_k=5)
        
        # Display results
        self.results_text.insert(tk.END, f"Top 5 matches for query: '{query}'\n")
        self.results_text.insert(tk.END, "-" * 50 + "\n")
        
        self.match_files = []
        for i, (file_path, score) in enumerate(matches):
            file_name = os.path.basename(file_path)
            self.results_text.insert(tk.END, f"{i+1}. {file_name} (score: {score:.4f})\n")
            self.match_files.append(file_path)
        
        self.results_text.config(state=tk.DISABLED)
        
        # Visualize top match
        if matches:
            top_match = matches[0][0]
            self.current_file_var.set(f"Selected: {os.path.basename(top_match)}")
            self.visualize_audio(top_match)
            self.play_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.current_file = top_match
        
        self.status_var.set(f"Found {len(matches)} matches for '{query}'")
    
    def visualize_audio(self, file_path):
        """Visualize the audio waveform and spectrogram"""
        import librosa
        import numpy as np
        
        # Clear previous plot
        self.fig.clear()
        
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Waveform
        ax1 = self.fig.add_subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title('Waveform')
        
        # Spectrogram
        ax2 = self.fig.add_subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2)
        self.fig.colorbar(img, ax=ax2, format='%+2.0f dB')
        ax2.set_title('Spectrogram')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def play_sound(self):
        """Play the selected sound"""
        if hasattr(self, 'current_file'):
            pygame.mixer.music.load(self.current_file)
            pygame.mixer.music.play()
            self.status_var.set(f"Playing: {os.path.basename(self.current_file)}")
    
    def stop_sound(self):
        """Stop the currently playing sound"""
        pygame.mixer.music.stop()
        self.status_var.set("Playback stopped")

def main():
    parser = argparse.ArgumentParser(description='CLAP Sound Matcher Demo')
    parser.add_argument('--audio_dir', type=str, default='data/ESC-50-master/audio',
                        help='Directory containing audio files')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA for CLAP model')
    
    args = parser.parse_args()
    
    # Create tkinter root
    root = tk.Tk()
    
    # Create app
    app = SoundMatcherApp(root, args.audio_dir, use_cuda=args.use_cuda)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
