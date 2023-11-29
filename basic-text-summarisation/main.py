import batch_processing as bp
from rouge import Rouge


def calculate_rouge_scores(gen_summaries, ref_summaries):
    """
    Calculates ROUGE scores for a set of generated summaries against reference summaries.

    Args:
    gen_summaries (list): A list of generated summaries.
    ref_summaries (list): A list of reference summaries.

    Returns:
    dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    rouge = Rouge()
    scores = rouge.get_scores(gen_summaries, ref_summaries, avg=True)
    return scores


# Specify the directory containing story files
directory = '/home/rajagopalmenon.v/cnn/stories'

# Process stories to get generated and reference summaries
generated_summaries, reference_summaries = bp.process_stories(directory, num_sentences=5)

# Calculate ROUGE scores for the summaries
rouge_scores = calculate_rouge_scores(generated_summaries, reference_summaries)
print("ROUGE Scores:", rouge_scores)

# Print the first 5 generated summaries for demonstration
for summary in generated_summaries[:5]:
    print(summary)
    print('---')
