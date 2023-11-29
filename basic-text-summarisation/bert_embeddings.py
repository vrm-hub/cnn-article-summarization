import torch


def get_sentence_embeddings(chunk, tokenizer, model, device):
    """
    Generates sentence embeddings for a given text chunk using a BERT model.

    Args:
    chunk (torch.Tensor): A tensor representing tokenized sentences.
    tokenizer (BertTokenizer): Tokenizer used for encoding text.
    model (BertModel): Pretrained BERT model for generating embeddings.
    device (torch.device): The device (CPU or GPU) used for computations.

    Returns:
    list: A list containing the sentence embeddings.
    """
    embeddings = []

    with torch.no_grad():
        if chunk.dim() == 1:
            # Adding batch dimension if missing
            chunk = chunk.unsqueeze(0)

        # Limit the sequence length to 512 tokens for BERT
        if chunk.size(1) > 512:
            chunk = chunk[:, :512]

        # Generate attention mask (1 for tokens, 0 for padding)
        attention_mask = (chunk != tokenizer.pad_token_id).long()

        # Move tensors to the specified device
        chunk = chunk.to(device)
        attention_mask = attention_mask.to(device)

        # Feed the chunk through the BERT model
        outputs = model(chunk, attention_mask=attention_mask)

        # Extract and process the output to get sentence embeddings
        # Mean pooling is used here to get a single vector for each input chunk
        sentence_embedding = outputs[0].mean(1).squeeze().numpy()
        embeddings.append(sentence_embedding)

    return embeddings
