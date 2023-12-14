import gdown
import shutil
import os
from pathlib import Path


URL_LINKS = {
    'rawnet2': 'https://drive.google.com/uc?id=182HC2ZKUY45Gsjj_6SUpA9WuZ1p-0P6g'
}

def main():
    dir = Path(__file__).absolute().resolve().parent.parent
    for name in URL_LINKS:
        checkpoint_dir = dir / 'saved' / 'models' / 'checkpoints' / name
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        zip_pth = checkpoint_dir / (name + '.zip')
        model_pth = checkpoint_dir / 'model_weights.pth'
        if not model_pth.exists():
            gdown.download(URL_LINKS[name], str(zip_pth), quiet=False)
            shutil.unpack_archive(str(zip_pth), str(checkpoint_dir), "zip")
            os.remove(zip_pth)

if __name__ == "__main__":
    main()