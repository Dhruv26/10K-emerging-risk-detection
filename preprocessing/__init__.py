import logging


logging_format = '%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:%(' \
                 'lineno)d — %(message)s'
logging.basicConfig(
    filename='risk_extractor.log',
    format=logging_format
)
