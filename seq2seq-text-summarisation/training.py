import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

def create_datasets(train_inputs, train_targets, val_inputs, val_targets, test_inputs, test_targets):
    """Creates datasets."""
    train_dataset = TensorDataset(train_inputs['input_ids'], train_targets['input_ids'])
    val_dataset = TensorDataset(val_inputs['input_ids'], val_targets['input_ids'])
    test_dataset = TensorDataset(test_inputs['input_ids'], test_targets['input_ids'])
    return train_dataset, val_dataset, test_dataset

def save_encoder_output(encoder, train_inputs, file_path):
    """Saves encoder output."""
    encoder.eval()
    with torch.no_grad():
        encoder_output, _, _ = encoder(train_inputs['input_ids'])
    torch.save(encoder_output, file_path)

def train_model(encoder, decoder, train_loader, optimizer, criterion, num_epochs):
    """Trains the model."""
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, labels = batch
            encoder_output, _, _ = encoder(input_ids)
            decoder_input = labels[:, :-1]
            decoder_output, _, _ = decoder(decoder_input, _, _)

            loss = criterion(decoder_output.view(-1, decoder_output.shape[-1]), labels[:, 1:].contiguous().view(-1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {total_loss / len(train_loader)}')

def evaluate_model(encoder, decoder, val_loader, criterion):
    """Evaluates the model."""
    encoder.eval()
    decoder.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            encoder_output, _, _ = encoder(input_ids)
            decoder_input = labels[:, :-1]
            decoder_output, _, _ = decoder(decoder_input, _, _)

            val_loss = criterion(decoder_output.view(-1, decoder_output.shape[-1]), labels[:, 1:].contiguous().view(-1))
            total_val_loss += val_loss.item()

    print(f'Validation Loss: {total_val_loss / len(val_loader)}')

def test_model(encoder, decoder, test_loader, criterion):
    """Tests the model."""
    encoder.eval()
    decoder.eval()
    total_test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            input_ids, labels = batch
            encoder_output, _, _ = encoder(input_ids)
            decoder_input = labels[:, :-1]
            decoder_output, _, _ = decoder(decoder_input, _, _)

            test_loss = criterion(decoder_output.view(-1, decoder_output.shape[-1]), labels[:, 1:].contiguous().view(-1))
            total_test_loss += test_loss.item()

    print(f'Test Loss: {total_test_loss / len(test_loader)}')
