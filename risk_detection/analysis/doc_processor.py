from typing import List, Union

import spacy
from spacy.tokens import Token, Span

spacy.prefer_gpu()
_nlp = spacy.load('en_core_web_lg')
_nlp.disable_pipes(['ner'])


class RiskSectionCleaner:
    def __init__(self, drop_stops: bool = True, lemmatize: bool = True):
        self.drop_stops = drop_stops
        self.lemmatize = lemmatize

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
        elif cleaned[0].text.lower() == 'company':
            cleaned = cleaned[1:]
        return cleaned.lemma_.lower()
