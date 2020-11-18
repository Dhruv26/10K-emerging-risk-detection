import json
import os
import sys
import re

import pandas as pd
from bs4 import BeautifulSoup

from config import Config

sys.setrecursionlimit(10**5)


def extract_risk_section_from_report(raw_10k):
    """
    Extracts risk section from 10K SEC reports.

    :param raw_10k: The raw 10K report
    :return: Extracted risk section
    :raises: RiskSectionNotFound error if it cannot
             find the risk section in the report
    """
    soup = BeautifulSoup(raw_10k, 'lxml')
    types = soup.find(
        lambda x: x.name == 'type' and x.get_text().startswith('10-K')
    )

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
    return risk_soup.get_text()


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
    with open(os.path.join(Config.data_path(), 'sample_report_1.txt'),
              'r') as f:
        doc = f.read()

    ex = extract_risk_section_from_report(doc)
