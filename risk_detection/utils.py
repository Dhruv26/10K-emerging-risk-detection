import os
from itertools import islice
from pathlib import Path
from typing import List, Iterable, Tuple

import pandas as pd
from pkg_resources import resource_stream

from config import Config


def get_immediate_subdirectories(dire) -> Tuple[str]:
    return tuple(f.name for f in os.scandir(dire) if f.is_dir())


all_ciks = get_immediate_subdirectories(Config.risk_dir())


def get_company_industry_mapping() -> pd.DataFrame:
    csv_stream = resource_stream(
        'risk_detection.analysis', os.path.join('static', 'cik_industry.csv')
    )
    return pd.read_csv(csv_stream)


def get_sik_industry_name_mapping() -> pd.DataFrame:
    csv_stream = resource_stream(
        'risk_detection.analysis',
        os.path.join('static', 'sic_industry_name.csv')
    )
    return pd.read_csv(csv_stream)


def get_risk_filenames() -> List[Path]:
    risk_dir = Path(Config.risk_dir())
    return list(risk_dir.rglob('*.txt'))


def get_risk_filenames_for_ciks(ciks: Iterable[int] = all_ciks) -> List[Path]:
    filenames = list()
    for cik in ciks:
        cik_dir = os.path.join(Config.risk_dir(), str(cik))
        files = list(Path(cik_dir).glob('*.txt'))
        if files:
            filenames.extend(files)
    return filenames


def get_risk_section_sentiment_files() -> List[Path]:
    risk_sentiment_files = Path(Config.risk_sentiment_dir()).rglob('*.pickle')
    return list(risk_sentiment_files)


def create_dir_if_not_exists(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_name_without_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def get_word_sentiment_df() -> pd.DataFrame:
    csv_stream = resource_stream(
        'risk_detection.analysis',
        os.path.join('static', 'LoughranMcDonald_MasterDictionary_2018.csv')
    )
    return pd.read_csv(csv_stream)


def window(seq, n=2):
    """
    Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator
    """
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
