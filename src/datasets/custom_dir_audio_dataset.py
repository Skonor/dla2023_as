import logging
from pathlib import Path
from torch.utils.data import Dataset
import torchaudio

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(Dataset):
    def __init__(self, audio_dir, target_sr=16000, max_len=None, limit=None, *args, **kwargs):
        super().__init__()
        data = []
        for path in Path(audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path"] = str(path)
            if len(entry) > 0:
                data.append(entry)
            if limit is not None:
                data = data[:limit]

        self.data = data
        self.max_len = max_len
        self.target_sr = target_sr
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        audio_path = self.data[index]["path"]
        audio_tensor, sr = torchaudio.load(audio_path)
        audio_tensor = audio_tensor[0:1, :]
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, self.target_sr)
        if self.max_len is not None:
            audio_tensor = audio_tensor[:, :self.max_len]

        return {
            "audio": audio_tensor,
            "is_spoofed": -1,
            "audio_path": audio_path
        }