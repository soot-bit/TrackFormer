from pathlib import Path
import os 
import pandas as pd

class EventManager:
    def __init__(self, folder="csv"):
        self.folder = Path.cwd() / folder
        self.file_types = ["spacepoint.csv", "particles.csv", "parameters.csv", "tracks.csv"]

    def get_event(self, event):
        event_files = event_files = [self.folder / f"{event}-{file}" for file in self.file_types]
        for file in event_files:
            if not file.exists():
                raise FileNotFoundError(f"File not found: {file}")
        data_frames = [pd.read_csv(file) for file in event_files]
        return data_frames