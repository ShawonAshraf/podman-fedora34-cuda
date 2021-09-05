import torch
from torch.utils.data import Dataset


class PolarityReviewDataset(Dataset):

    def __init__(self, reviews, labels, tokenizer, MAX_LEN):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]

        # encode review text
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.MAX_LEN,
            truncation=True,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "label": torch.tensor(label)
        }
