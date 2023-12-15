
## Overview

This repo contains RawNet2 implemantation and training framework on ASVspoof2019 dataset (LA part) as well as scripts for ablations.


## Installation guide

```shell
pip install -r ./requirements.txt
```

To load checkpoints run:
```shell
python scripts/load_chheckpoints.py
```


## Training
To reproduce training do the following (All training was done on kaggl)

1. Train RawNet2 for 50k steps

```shell
python train.py -c src/configs/RawNet2_configs/train.json
```

## Evaluation

For evaluating model on a custom dir dataset do the following:


1. (Optional) Load checkpoint from training:
```shell
python scripts/load_chheckpoints.py
```
This will create rawnet2 directory in saved/models/checkpoints contaning model weigths file and training config

You can skip this step if you are using you own model

2. Run test.py (here for example I used data from test_audio/kaggle_test):
```shell
python test.py -r saved/checkpoints/rawnet2/model_weights.pth -t test_audio/kaggle_test -o test_audio/kaggle_test/model_pred.json -b 1
```

This will create model_pred.json file containing probabilities of each utterance being spoofed.


## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
