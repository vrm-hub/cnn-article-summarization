from torch.utils.data import DataLoader
from data_processing import read_csv, save_tokenized_data, load_tokenized_data, tokenize_data, create_datasets
from custom_models import CustomEncoder, CustomDecoder
from training import save_encoder_output, train_model, evaluate_model, test_model
import torch.optim as optim
import torch.nn as nn
from transformers import GPT2Tokenizer

# Sample data loading
train_df = read_csv('train.csv')
val_df = read_csv('validation.csv')
test_df = read_csv('test.csv')

# Tokenize and save data
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
train_inputs, train_targets = tokenize_data(tokenizer, train_df)
val_inputs, val_targets = tokenize_data(tokenizer, val_df)
test_inputs, test_targets = tokenize_data(tokenizer, test_df)

save_tokenized_data(train_inputs, train_targets, 'train')
save_tokenized_data(val_inputs, val_targets, 'val')
save_tokenized_data(test_inputs, test_targets, 'test')

# Load tokenized data
train_inputs, train_targets = load_tokenized_data('train')
val_inputs, val_targets = load_tokenized_data('val')
test_inputs, test_targets = load_tokenized_data('test')

# Create datasets
train_dataset, val_dataset, test_dataset = create_datasets(train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets)

# Instantiate custom encoder and decoder
input_dim = len(tokenizer.get_vocab())
output_dim = len(tokenizer.get_vocab())
embedding_dim = 256
hidden_dim = 512

encoder = CustomEncoder(input_dim, embedding_dim, hidden_dim)
decoder = CustomDecoder(output_dim, embedding_dim, hidden_dim)

# Use larger batch size for training and validation
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Save encoder output after encoding
encoder.eval()
save_encoder_output(encoder, train_inputs, 'encoder_output.pt')

# Define optimizer and loss function
optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 3  # Adjust as needed
train_model(encoder, decoder, train_loader, optimizer, criterion, num_epochs)

# Evaluation loop
evaluate_model(encoder, decoder, val_loader, criterion)

# Test the model
test_model(encoder, decoder, test_loader, criterion)
