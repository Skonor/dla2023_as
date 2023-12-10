from pathlib import Path

import torchaudio
from torch.utils.data import Dataset



class ASVspoof2019LA(Dataset):
    def __init__(self, data_dir, part='train', limit=None, max_len=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data_dir = Path(data_dir)
        self.max_len = max_len

        if part == "train":
            protocol_path = data_dir / 'LA' / 'LA' / 'ASVspoof2019_LA_cm_protocols' / f'ASVspoof2019.LA.cm.{part}.trn.txt'
        else:
            protocol_path = data_dir / 'LA' / 'LA' / 'ASVspoof2019_LA_cm_protocols' / f'ASVspoof2019.LA.cm.{part}.trl.txt'

        self.flac_dir = data_dir / 'LA' / 'LA' / f'ASVspoof2019_LA_{part}.flac'

        with open(protocol_path, 'r') as f:
            self.index = f.read().splitlines()

        if limit is not None:
            self.index = self.index[:limit]



    def __len__(self):
        return len(self.index)


    def _getitem__(self, item):
        _, utterenceID, _, _, IsSpoofed = self.index[item].split()
        audio_path = self.flac_dir /  (utterenceID + '.flac')

        audio_tensor, sr = torchaudio.load(audio_path)
        audio_tensor = audio_tensor[0:1, :]
        if self.max_len is not None:
            audio_tensor = audio_tensor[:, :self.max_len]

        if IsSpoofed == 'spoof':
            is_spoofed = 1
        else:
            is_spoofed = 0

        return {
            "audio": audio_tensor,
            "is_spoofed": is_spoofed
        }
