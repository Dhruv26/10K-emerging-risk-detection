import logging
import os

from config import Config

logging_format = '%(asctime)s — %(levelname)s — %(funcName)s:%(' \
                 'lineno)d — %(message)s'
logging.basicConfig(
    filename=os.path.join(Config.log_dir(), 'risk_extractor.log'),
    format=logging_format,
    level=logging.DEBUG
)
