import os
from glob import glob
from datetime import datetime
from typing import List
from pathlib import Path
from collections import Counter
import random
from scipy.spatial.distance import cosine

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

from wordcloud_generator import create_wordcloud
from risk_detection.analysis.keyword_extraction import get_keywords_for, generate_keywords
from risk_detection.preprocessing.report_parser import (
    report_info_from_risk_path, ReportInfo
)
from config import Config
from risk_detection.utils import get_immediate_subdirectories


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

keyword_file_lookup = dict()


def _get_report_infos(files) -> List[ReportInfo]:
    report_infos = list()
    for report_file in files:
        file_path = Path(report_file)
        report_info = report_info_from_risk_path(file_path)
        keyword_file_lookup[report_info] = file_path
        report_infos.append(report_info)

    return report_infos


keywords_dir = Config.text_rank_keywords_dir()
ciks = {cik: _get_report_infos(glob(os.path.join(keywords_dir, cik, '*.txt')))
        for cik in get_immediate_subdirectories(keywords_dir)}


NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(
                        dbc.NavbarBrand("Emerging Risk Detection",
                                        className="ml-2")
                    ),
                ],
                align='center',
                no_gutters=True,
            ),
            href='https://github.com/Dhruv26/10K-emerging-risk-detection',
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

WORDCLOUD_PLOTS = [
    dbc.CardHeader(html.H5("Risks for a company")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="loading-bigrams-comps",
                children=[
                    dbc.Alert(
                        "Something's gone wrong! Give us a moment, but try loading this page again if problem persists.",
                        id="no-data-alert-bigrams_comp",
                        color="warning",
                        style={"display": "none"},
                    ),
                    dbc.Row(
                        [
                            dbc.Col(html.P("Choose two companies to compare:"),
                                    md=12),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="cik",
                                        options=[
                                            {"label": i, "value": i}
                                            for i in ciks.keys()
                                        ],
                                        value="1000045",
                                    )
                                ],
                                md=6,
                            ),
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id="year",
                                    )
                                ],
                                md=6,
                            ),
                        ]
                    ),
                    html.Div(
                        html.Img(id="image_wc", height=600, width=1000),
                        style={'textAlign': 'center'}
                    ),
                    # html.Img(id="image_wc_neg")
                ],
                type="default",
            ),
            dbc.Row([
                dbc.Col([dcc.Dropdown(id='keywords')])
            ]),
            dash_table.DataTable(
                id='sentence_df', columns=[], data=[],
                style_data={'whiteSpace': 'normal', 'height': 'auto'},
                style_cell={'textAlign': 'left'}
            ),
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(dbc.Card(WORDCLOUD_PLOTS))],
                style={"marginTop": 30, 'marginBottom': 50}),
        # dbc.Row([dbc.Col(dbc.Card(TOP_BIGRAM_PLOT)),], style={"marginTop": 30}),
        # dbc.Row(
        #     [
        #         dbc.Col(LEFT_COLUMN, md=4, align="center"),
        #         dbc.Col(dbc.Card(TOP_BANKS_PLOT), md=8),
        #     ],
        #     style={"marginTop": 30},
        # ),
        # dbc.Row([dbc.Col([dbc.Card(LDA_PLOTS)])], style={"marginTop": 50}),
    ],
    className="mt-12",
)


app.layout = html.Div(children=[NAVBAR, BODY])


@app.callback(Output('image_wc', 'src'), [Input('year', 'value')])
def plot_wordcloud(doc_id):
    keyword_path = keyword_file_lookup[ReportInfo.from_doc_id(doc_id)]
    keywords = get_keywords_for(keyword_path)
    return create_wordcloud(keywords.keywords)


# @app.callback(Output('image_wc_neg', 'src'), [Input('image_wc_neg', 'id')])
# def plot_wordcloud(b):
#     return create_wordcloud(neg_keys.get_negative_keywords())

@app.callback(Output('keywords', 'options'), Input('year', 'value'))
def set_neg_keywords_options(doc_id):
    keyword_path = keyword_file_lookup[ReportInfo.from_doc_id(doc_id)]
    keywords = get_keywords_for(keyword_path)
    return [{'label': key, 'value': key}
            for key in keywords.get_negative_keywords()]


@app.callback(Output('keywords', 'value'), Input('keywords', 'options'))
def set_neg_keywords_value(options):
    return options[0]['value']


@app.callback([Output('sentence_df', 'columns'),
               Output('sentence_df', 'data')],
              [Input('year', 'value'), Input('keywords', 'value')])
def populate_sentences_df(doc_id, keyword):
    keyword_path = keyword_file_lookup[ReportInfo.from_doc_id(doc_id)]
    keywords = get_keywords_for(keyword_path)

    try:
        sentence_df = keywords.neg_keywords[keyword]
        return ([{"name": i, "id": i} for i in sentence_df.columns],
                sentence_df.to_dict(orient='records'))
    except KeyError:
        return [], []


@app.callback(Output('year', 'options'), Input('cik', 'value'))
def set_cities_options(selected_cik):
    def get_str(datetime_obj: datetime) -> str:
        return datetime_obj.strftime('%Y:%m:%d')

    return [{'label': f'{get_str(i.start_date)}-{get_str(i.end_date)}',
             'value': i.get_document_id()}
            for i in ciks[selected_cik]]


@app.callback(Output('year', 'value'), Input('cik', 'value'))
def set_cities_value(available_options):
    report_info = ciks[available_options][0]
    return report_info.get_document_id()


if __name__ == '__main__':
    app.run_server(debug=True)
