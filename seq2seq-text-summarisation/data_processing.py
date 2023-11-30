import pandas as pd
import torch
from transformers import GPT2Tokenizer
from torch.utils.data import TensorDataset

def read_csv(file_path):
    """Reads CSV file."""
    return pd.read_csv(file_path)

def tokenize_data(tokenizer, data):
    """Tokenizes input and target texts using GPT-2 tokenizer."""
    tokenizer.pad_token = tokenizer.eos_token
    input_texts, target_texts = data['article'].astype(str).tolist(), data['highlights'].astype(str).tolist()
    input_encodings = tokenizer(input_texts, truncation=True, padding=True, return_tensors='pt', max_length=512)
    target_encodings = tokenizer(target_texts, truncation=True, padding=True, return_tensors='pt', max_length=512)
    return input_encodings, target_encodings

def save_tokenized_data(inputs, targets, file_prefix):
    """Saves tokenized data to files."""
    torch.save(inputs, f'{file_prefix}_inputs.pt')
    torch.save(targets, f'{file_prefix}_targets.pt')

def load_tokenized_data(file_prefix):
    """Loads tokenized data from files."""
    inputs = torch.load(f'{file_prefix}_inputs.pt')
    targets = torch.load(f'{file_prefix}_targets.pt')
    return inputs, targets

def create_datasets(train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets):
    """Creates datasets."""
    train_dataset = TensorDataset(train_inputs['input_ids'], train_targets['input_ids'])
    val_dataset = TensorDataset(val_inputs['input_ids'], val_targets['input_ids'])
    test_dataset = TensorDataset(test_inputs['input_ids'], test_targets['input_ids'])
    return train_dataset, val_dataset, test_dataset
