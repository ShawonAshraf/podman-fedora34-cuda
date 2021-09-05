
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from classifier import SentiBERT
from dataset import PolarityReviewDataset

from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split

from corpus import *
from tqdm import tqdm
import os

batch_size = 16
model_name = "bert-base-cased"

# the sentences in this dataset are paragraph sized and
# bert has a hard limit of 512 tokens per sentence.
# so the model will perform pretty bad on a test set.
# there are numerous papers on this on why that limitation
# exists for transformer based models. or if you're curious enough
# read the original paper on BERT and try figuring it out ;)
# https://arxiv.org/abs/1706.03762
# https://arxiv.org/abs/2102.11174
# https://arxiv.org/abs/1810.04805
# https://proceedings.neurips.cc/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf
# https://aclanthology.org/D19-5821/
# https://openreview.net/forum?id=SklnVAEFDB

# unless you have more than 8GB VRAM, stick with 128.
# Even with 512, the max size the
# results aren't promising. You can't bypass the laws of mathematics.
# It's not a south indian movie.
MAX_LEN = 128

download_and_unzip()

reviews = []
labels = []

# we can't use the previous tokenizers here
# idx 0 -> neg, 1 -> pos
for idx, cat in enumerate(catgeories):
    path = os.path.join(corpus_root, cat)
    texts = read_text_files(path)

    for i in tqdm(range(len(texts)), desc="prepare_corpus"):
        text = texts[i]
        reviews.append(text)
        labels.append(idx)

tokenizer = AutoTokenizer.from_pretrained(model_name)
x_train, x_test, y_train, y_test = train_test_split(
    reviews, labels, random_state=42, train_size=0.8
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.8, random_state=42)


training_dataset = PolarityReviewDataset(x_train, y_train, tokenizer, MAX_LEN)
val_dataset = PolarityReviewDataset(x_val, y_val, tokenizer, MAX_LEN)
test_dataset = PolarityReviewDataset(x_test, y_test, tokenizer, MAX_LEN)


train_loader = DataLoader(
    training_dataset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


model = SentiBERT(model_name)
trainer = pl.Trainer(gpus=1, max_epochs=2)
trainer.fit(model, train_loader, val_loader)

# test the model
trainer.test(model, test_loader)
