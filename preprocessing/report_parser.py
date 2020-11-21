import json
import os
import re
from io import BytesIO
from unicodedata import normalize

import pandas as pd
from bs4 import BeautifulSoup
from pkg_resources import resource_string

from config import Config


def _read_risk_start_end_data():
    risk_start_end_csv = resource_string(
        'preprocessing', os.path.join('resources', 'RiskFactors_StartEnd.csv')
    )
    cols = ['cik', 'documentlinktext', 'sectionstart', 'sectionend']
    df = pd.read_csv(BytesIO(risk_start_end_csv), usecols=cols)
    df['documentlinktext'] = df['documentlinktext'].str.split('/').str[-1]
    return df


# TODO: Remove if not required
_risk_start_end_df = _read_risk_start_end_data()


def extract_risk_section_from_report(raw_10k):
    """
    Extracts risk section from 10K SEC reports.

    :param raw_10k: The raw 10K report
    :return: Extracted risk section
    :raises: RiskSectionNotFound error if it cannot
             find the risk section in the report
    """
    types = _get_risk_document(raw_10k)

    raw_report = str(types)
    pos_data = _get_risk_section_from_soup(raw_report)
    raw_risk = raw_report[
               pos_data['start'].loc['item1a']:pos_data['start'].loc['item1b']
               ]
    risk_soup = BeautifulSoup(raw_risk, 'lxml')

    # The first p tag contains the Risk Section title. Hence we remove it.
    risk_title_tag = risk_soup.find('p')
    if risk_title_tag:
        risk_title_tag.decompose()
    return normalize("NFKC", risk_soup.get_text(strip=True))


def _get_risk_document(raw_10k):
    soup = BeautifulSoup(raw_10k, 'lxml')
    return soup.find(
        lambda x: x.name == 'type' and x.get_text().startswith('10-K')
    )


def _get_risk_section_from_soup(raw_report):
    # Regex to find start of section 1A and 1B
    risk_section_regex = re.compile(
        r'(>Item(\s|&#160;|&nbsp;)+(1A|1B)\.?)|(ITEM(\s)+(1A|1B))'
    )
    risk_section_matches = [
        {
            'item': x.group(), 'start': x.start(), 'end': x.end()
        }
        for x in (risk_section_regex.finditer(raw_report))
    ]
    if not risk_section_matches:
        raise RiskSectionNotFound('Risk section not found.')

    match_df = pd.read_json(json.dumps(risk_section_matches))
    match_df['item'] = match_df.item.str.lower()
    return _create_matches_df(match_df)


def _create_matches_df(match_df):
    # Get rid of unnecessary characters from the dataframe
    match_df.replace('&#160;', ' ', regex=True, inplace=True)
    match_df.replace('&nbsp;', ' ', regex=True, inplace=True)
    match_df.replace(' ', '', regex=True, inplace=True)
    match_df.replace('\.', '', regex=True, inplace=True)
    match_df.replace('>', '', regex=True, inplace=True)

    _validate_match_df(match_df)
    return (match_df.sort_values('start', ascending=True)
            .drop_duplicates(subset=['item'], keep='last')
            .set_index('item'))


def _validate_match_df(match_df):
    unique_items = match_df['item'].unique()
    if 'item1a' not in unique_items:
        raise RiskSectionNotFound('Item 1A not found.')
    if 'item1b' not in unique_items:
        raise RiskSectionNotFound('Item 1B not found.')


class RiskSectionNotFound(Exception):
    def __init__(self, message):
        super().__init__(message)


if __name__ == '__main__':
    with open(os.path.join(Config.data_dir(), 'sample_report.txt'),
              'r') as f:
        doc = f.read()

    ex = extract_risk_section_from_report(doc)
