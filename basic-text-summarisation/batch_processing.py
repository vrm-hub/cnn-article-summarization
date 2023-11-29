import os
import time
import preprocessing as pp
import bert_embeddings as be
import summarization as sm
import utility as ut


def process_stories(directory, num_sentences=3, limit=10000):
    """
    Processes stories in the specified directory to generate summaries. Skips already processed stories.

    Args:
    directory (str): Directory containing story files.
    num_sentences (int): Number of sentences to include in each summary.
    limit (int): Maximum number of files to process.

    Returns:
    tuple: Tuple containing two lists - generated summaries and reference summaries.
    """
    generated_summaries = []
    reference_summaries = []
    count = 0

    # Directory for saving processed data
    processed_data_dir = '../processed_data'
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    for filename in os.listdir(directory):
        start = time.time()
        if filename.endswith('.story') and count < limit:
            processed_data_file = os.path.join(processed_data_dir, f'{filename}_processed.pkl')

            # Check if the story has already been processed
            if os.path.exists(processed_data_file):
                data = ut.load_data(processed_data_file)
                generated_summaries.append(data['generated_summary'])
                reference_summaries.append(data['reference_summary'])
            else:
                # Process the story
                story_path = os.path.join(directory, filename)
                story_text, reference_summary, tokenizer, model, device = pp.preprocess_story_file(story_path)
                sentences, tokenized_chunks = pp.preprocess_text(story_text)

                all_embeddings = [embedding for chunk in tokenized_chunks
                                  for embedding in be.get_sentence_embeddings(chunk, tokenizer, model, device)]

                summary = sm.summarize_article(sentences, all_embeddings, num_sentences)

                # Save the summary and reference summary with proper labels
                save_data = {
                    'generated_summary': summary,
                    'reference_summary': reference_summary
                }
                ut.save_data(save_data, processed_data_file)

                generated_summaries.append(summary)
                reference_summaries.append(reference_summary)

            print(f"Processed file {count} - {filename} - time taken: {time.time() - start}")
            count += 1

    return generated_summaries, reference_summaries
