from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent

def remove_link(text):
    return re.sub(r'https?:\/\/\S*', '', text, flags=re.MULTILINE)


def remove_username(text):
    return re.sub('@[\w]+', '', text)


def remove_retweet(text):
    return text.replace("RT : ", "")


def remove_n(text):
    return text.replace("\n", "").strip()


def augment_text(text):
    """
    functie om ongewenste combinaties van tekens te verwijderen uit de text

    :param text: text die wordt aangepast

    :return: text met verwijderde tekens
    """
    return remove_n(remove_retweet(remove_username(remove_link(text))))


class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


tokenizer = RobertaTokenizerFast.from_pretrained(f"{BASE_DIR}/models/tokenizer", model_max_length=512,
                                                 return_tensors='pt', num_labels=3)

def mod_predict(model, text):
    model = RobertaForSequenceClassification.from_pretrained(f"{BASE_DIR}/models/RobBERT_{model}")
    text = [augment_text(text)]
    trainer = Trainer(model=model)
    tokenized_text = tokenizer(text, truncation=True, padding=True)
    text_dataset = SimpleDataset(tokenized_text)
    predictions = trainer.predict(text_dataset)
    predictions_in_probabilities = torch.nn.functional.softmax(torch.Tensor(predictions[0]), dim=1)
    return int(np.argmax(predictions_in_probabilities))
