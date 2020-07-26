import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_TSV = "data/train.tsv"
TEST_TSV = "data/test.tsv"


def load_data(test_size=0.2):
    raw_data_train = pd.read_csv(TRAIN_TSV, sep="\t")
    data_test = pd.read_csv(TEST_TSV, sep="\t")
    data_train, data_val = train_test_split(raw_data_train, test_size=test_size)

    X_train = data_train["Phrase"].values
    X_val = data_val["Phrase"].values
    X_test = data_test["Phrase"].values
    y_train = data_train["Sentiment"].values
    y_val = data_val["Sentiment"].values

    data_train = data_train.sort_values("PhraseId")

    sentences_train = data_train.drop_duplicates("SentenceId")["Phrase"]

    return X_train, X_val, X_test, y_train, y_val, sentences_train
