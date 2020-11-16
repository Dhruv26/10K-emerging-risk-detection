import json
import re

import pandas as pd
from bs4 import BeautifulSoup


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
    risk_title_tag.decompose()
    return risk_soup.get_text()


def _get_risk_section_from_soup(raw_report):
    # Regex to find start of section 1A and 1B
    risk_section_regex = re.compile(
        r'(>Item(\s|&#160;|&nbsp;)(1A|1B)\.{0,1})|(ITEM\s(1A|1B))'
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

    return (match_df.sort_values('start', ascending=True)
            .drop_duplicates(subset=['item'], keep='last')
            .set_index('item'))


class RiskSectionNotFound(Exception):
    def __init__(self, message):
        super().__init__(message)
