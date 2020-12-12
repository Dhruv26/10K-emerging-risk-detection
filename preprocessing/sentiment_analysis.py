from finbert.finbert import predict as _predict
from pytorch_pretrained_bert import BertForSequenceClassification

from config import Config

_model_path = Config.finBERT_model_dir()
_model = BertForSequenceClassification.from_pretrained(
    _model_path, num_labels=3, cache_dir=True
)


def get_sentiment(text):
    """
    Predicts the sentiment of each sentence in text. Relies BERT
    on the model produced by https://github.com/ProsusAI/finBERT.

    :param text: Can be a paragraph or a single sentence.
        A paragraph will be tokenized.
    :return: Pandas dataframe containing each sentence in
        text along with info about it's sentiment.
    """
    return _predict(text, _model)
