import json
import re

import pandas as pd
from bs4 import BeautifulSoup


def extract_risk_section_from_report(raw_10k):
    soup = BeautifulSoup(raw_10k, 'lxml')
    types = soup.find(
        lambda x: x.name == 'type' and x.get_text().startswith('10-K')
    )
    report_10k_soup = types.find('xbrl')

    pos_data = _get_risk_section_from_soup(report_10k_soup)
    raw_report = str(report_10k_soup)
    raw_risk = raw_report[
               pos_data['start'].loc['item1a']:pos_data['start'].loc['item1b']
               ]
    risk_soup = BeautifulSoup(raw_risk, 'lxml')

    risk_title_tag = risk_soup.find('p', text=re.compile('Item (1A|1a)'))
    risk_title_tag.decompose()
    return risk_soup.get_text()


def _get_risk_section_from_soup(report_10k_soup):
    raw_report = str(report_10k_soup)

    # Regex to find start of section 1A and 1B
    risk_section_regex = re.compile(
        r'(>Item(\s|&#160;|&nbsp;)(1A|1B)\.{0,1})|(ITEM\s(1A|1B))'
    )
    matches = json.dumps([
        {'item': x.group(), 'start': x.start(), 'end': x.end()}
        for x in (risk_section_regex.finditer(raw_report))
    ])
    match_df = pd.read_json(matches)
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
