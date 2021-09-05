import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from transformers import AutoModel

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

from sklearn.metrics import f1_score


class SentiBERT(LightningModule):
    def __init__(self, model_name):
        super(SentiBERT, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        # 768 for BERT hidden dimensions, 1 for binary classification
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        input_ids, attention_mask = x
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out = out.pooler_output

        out = self.linear(out)
        out = self.sigmoid(out)

        return out

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=2e-5)

    def training_step(self, batch, batch_idx):
        td = batch

        input_ids = td["input_ids"]
        attention_mask = td["attention_mask"]
        label = td["label"]

        out = self((input_ids, attention_mask))
        logits, _ = torch.max(out, dim=1)
        loss = self.loss_fn(logits, label.float())

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        td = batch

        input_ids = td["input_ids"]
        attention_mask = td["attention_mask"]
        label = td["label"]

        out = self((input_ids, attention_mask))
        logits, _ = torch.max(out, dim=1)
        loss = self.loss_fn(logits, label.float())

        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        td = batch

        input_ids = td["input_ids"]
        attention_mask = td["attention_mask"]
        label = td["label"]

        out = self((input_ids, attention_mask))
        logits, _ = torch.max(out, dim=1)

        logits = torch.round(logits.squeeze())

        # dear pytorch team, find a easier wrapper please!
        labels_numpy = label.cpu().detach().numpy()
        predicted_numpy = logits.cpu().detach().numpy()

        score = f1_score(labels_numpy, predicted_numpy)

        self.log(f"f1_score::batch{batch_idx}", score, prog_bar=True)
