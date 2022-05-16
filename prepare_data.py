from pathlib import Path
from typing import Tuple

import pandas as pd

DEFAULT_OUTPUT_PATH = Path("data") / "labelled_sentences.csv"

# Define default paths as found in the zip file downloaded from https://nlp.stanford.edu/sentiment/code.html
DEFAULT_ROOT_FOLDER = Path("stanfordSentimentTreebank")
DATASET_SENTENCES_PATH = DEFAULT_ROOT_FOLDER / "datasetSentences.txt"
DATASET_SPLIT_PATH = DEFAULT_ROOT_FOLDER / "datasetSplit.txt"
DICTIONARY_PATH = DEFAULT_ROOT_FOLDER / "dictionary.txt"
SENTIMENT_LABELS_PATH = DEFAULT_ROOT_FOLDER / "sentiment_labels.txt"


def read_files() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # datasetSentences.txt has columns (sentence_id, sentence) divided by a tab character (\t)
    sentences = pd.read_csv(DATASET_SENTENCES_PATH, sep="\t")
    # datasetSplit.txt has columns (sentence_index, splitset_label) separated by a comma character (,)
    # The splitset_label is either 	1 = train, 2 = test or 3 = dev. Let's replace it in the df for easier use.
    splits = pd.read_csv(DATASET_SPLIT_PATH, sep=",")
    splits.loc[:, "splitset_label"] = splits["splitset_label"].replace({1: "train", 2: "test", 3: "dev"})
    # dictionary.txt has columns corresponding to a phrase and a phrase ID, with no headers, separated by a pipe character (|)
    dictionary = pd.read_csv(DICTIONARY_PATH, sep="|", header=None).rename(columns={0: "phrase", 1: "phrase_id"})
    # sentiment_labels.txt contains our labels, where each phrase ID corresponds to a float value. They are separated by a pipe character (|)
    sentiments = pd.read_csv(SENTIMENT_LABELS_PATH, sep="|")

    return sentences, splits, dictionary, sentiments


def merge_and_save_data(output_path: Path = DEFAULT_OUTPUT_PATH):
    sentences, splits, dictionary, sentiments = read_files()

    # First, merge sentences with dictionary to retrieve the correct phrase_id for each sentence_id
    sentences_with_phrase_id = pd.merge(sentences, dictionary, left_on="sentence", right_on="phrase")

    # During this step we lose 569 sentences that were not able to be matched to a phrase
    # This happens primarily for UNICODE issues: sentences in datasetSentences.txt appear to have been badly encoded
    # compared to similar pharses in dictionary.txt. For example:
    # "257	JirÃ­ Hubac 's script is a gem ." compared to "Jirí Hubac 's script is a gem .|224463"
    print(f"Lost {len(sentences) - len(sentences_with_phrase_id)} sentences in merge operation")

    # Now we merge the sentences to the corresponding labels using the "phrase_id" attribute
    labelled_sentences = pd.merge(sentences_with_phrase_id, sentiments, left_on="phrase_id", right_on="phrase ids")

    # And finally we merge the split information and save
    splitted_labelled_sentences = pd.merge(labelled_sentences, splits, left_on="sentence_index", right_on="sentence_index")

    # Make output directory if it does not exist
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    (
        splitted_labelled_sentences.rename(columns={"sentiment values": "label", "splitset_label": "split"})
        [["sentence_index", "sentence", "label", "split"]]
    ).to_csv(output_path, index=False, header=True)


if __name__ == "__main__":
    merge_and_save_data()
