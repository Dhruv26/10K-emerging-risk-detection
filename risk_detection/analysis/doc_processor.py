import os
import hashlib
import pickle
from functools import wraps
from typing import List, Union

import spacy
from gensim.parsing.preprocessing import strip_tags
from gensim.utils import simple_preprocess
from spacy.tokens import Token, Span

from config import Config
from risk_detection.utils import create_dir_if_not_exists

spacy.prefer_gpu()
_nlp = spacy.load(Config.spacy_model())
_nlp.disable_pipes(['ner'])

_base_cache_dir = os.path.join(Config.cache_dir(), 'tokens')


def cache_tokens(dir_name):
    """
    Caches tokens created for a document.

    :param dir_name: Directory to cache data to
    """
    def decorator(func):
        cache_dir = os.path.join(_base_cache_dir, dir_name)
        create_dir_if_not_exists(cache_dir)

        @wraps(func)
        def wrapper(obj, text):
            hashh = hashlib.sha256(text.encode('utf-8')).hexdigest()
            file_path = os.path.join(cache_dir, f'{hashh}.pickle')
            try:
                # Get from cache
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                # Cache result
                result = func(obj, text)
                # with open(file_path, 'wb') as f:
                #     pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
                return result

        return wrapper
    return decorator


class RiskSectionCleaner:
    def __init__(self, drop_stops: bool = True, lemmatize: bool = True):
        # TODO: See caching behaviour
        self.drop_stops = drop_stops
        self.lemmatize = lemmatize

    @cache_tokens(dir_name='simple_tokens')
    def simple_tokenize(self, doc_text: str) -> List[str]:
        return simple_preprocess(strip_tags(doc_text), deacc=True)

    @cache_tokens(dir_name='noun_phrase_tokens')
    def tokenize(self, doc_text: str) -> List[str]:
        tokens = []

        doc = _nlp(doc_text)
        noun_chunks = doc.noun_chunks
        try:
            curr_chunk = next(noun_chunks)
        except StopIteration:
            # No noun chunks in the provided text.
            # Defaulting to simple tokens.
            return [self._get_token(tkn)
                    for tkn in doc if self._is_valid_token(tkn)]

        for tkn in doc:
            if curr_chunk:
                if tkn.i == curr_chunk.end - 1:
                    # Last token for the curr noun chunk.
                    # Add the noun chunk to the list of tokens.
                    sanitized_chunk = self._sanitize_noun_chunk(curr_chunk)
                    if sanitized_chunk:
                        tokens.append(sanitized_chunk)
                    try:
                        curr_chunk = next(noun_chunks)
                    except StopIteration:
                        curr_chunk = None

                    # Move to next token
                    continue
                elif curr_chunk.start <= tkn.i:
                    # Wait for curr noun chunk to end
                    continue

            if self._is_valid_token(tkn):
                tokens.append(self._get_token(tkn))

        return tokens

    def __call__(self, doc_text: str) -> List[str]:
        return self.tokenize(doc_text)

    def _is_valid_token(self, token: Token) -> bool:
        return token.is_alpha and not (self.drop_stops and token.is_stop)

    def _get_token(self, token: Union[Span, Token]) -> str:
        return (token.lemma_ if self.lemmatize else token.text).lower()

    @staticmethod
    def _sanitize_noun_chunk(noun_chunk: Span) -> str:
        cleaned = noun_chunk
        if noun_chunk[0].is_stop or noun_chunk[0].is_punct:
            cleaned = cleaned[1:]
        if cleaned[:2].text.lower() == 'company\'s':
            cleaned = cleaned[2:]
        elif len(cleaned) > 0 and cleaned[0].text.lower() == 'company':
            cleaned = cleaned[1:]
        return cleaned.lemma_.lower()
