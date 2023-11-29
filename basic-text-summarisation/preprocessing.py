import nltk
from nltk.tokenize import sent_tokenize
from transformers import BertTokenizer, BertModel
import torch

# Download NLTK punkt tokenizer for sentence tokenization
nltk.download('punkt')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model.to(device)


def preprocess_text(text, max_length=512):
    """
    Tokenizes text into segments suitable for BERT processing, with each segment
    containing a single sentence.

    Args:
    text (str): The text to be tokenized.
    max_length (int): Maximum length of a tokenized chunk.

    Returns:
    tuple: A tuple containing two elements:
        - List of sentences from the text.
        - List of tokenized chunks, each representing a single sentence as a PyTorch tensor.
    """
    sentences = sent_tokenize(text)
    tokenized_chunks = []

    for sentence in sentences:
        # Tokenize each sentence individually with special tokens
        tokenized_sentence = tokenizer.encode(sentence, add_special_tokens=True)

        # Truncate the sentence if it's too long
        if len(tokenized_sentence) > max_length:
            tokenized_sentence = tokenized_sentence[:max_length-1] + [tokenizer.sep_token_id]

        # Convert to tensor and add to the list
        tokenized_chunks.append(torch.tensor(tokenized_sentence, device=device))

    return sentences, tokenized_chunks


def preprocess_story_file(file_path):
    """
    Reads a story file and extracts the story text and reference summary.

    Args:
    file_path (str): Path to the story file.

    Returns:
    tuple: A tuple containing:
        - The story text.
        - The reference summary extracted from highlights.
        - BERT tokenizer, model, and computation device.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        story = file.read()
    story_text, highlights = story.split('@highlight', 1)
    story_text = story_text.strip()

    highlights_list = highlights.split('@highlight')
    reference_summary = ''
    for highlight in highlights_list:
        reference_summary += highlight.strip() + ' '

    return story_text, reference_summary.strip(), tokenizer, model, device
