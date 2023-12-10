from torch import nn
from torch.nn import Sequential

from src.base import BaseModel


class DummyModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=2, **batch):
        super().__init__(n_feats, n_class, **batch)
        self.net = Sequential(
            # people say it can aproximate any function...
            nn.Linear(in_features=1, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=2)
        )

    def forward(self, audio, **batch):
        x = self.net(audio.unsqueeze(-1))
        x = x[:, -1, :]
        return {"logits": x}

