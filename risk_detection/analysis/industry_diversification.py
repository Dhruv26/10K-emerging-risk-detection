import os

from top2vec import Top2Vec

from config import Config
from risk_detection.preprocessing.report_parser import (
    report_info_from_risk_path
)
from risk_detection.utils import (get_company_industry_mapping,
                                  get_sik_industry_name_mapping,
                                  get_risk_filenames)


def _get_noun_phrases(txt):
    pass


model_path = os.path.join(Config.top2vec_models_dir(),
                          'top2vec_model_deep_with_doc_ids')
model = Top2Vec.load(model_path)

cik_sic_df = get_company_industry_mapping()
sic_name_df = get_sik_industry_name_mapping()
risk_files = get_risk_filenames()
# TODO: Add diversification detection code

for risk_file in risk_files:
    report_info = report_info_from_risk_path(risk_file)
    doc_id = report_info.get_document_id()
    topic = model.get_documents_topics(doc_ids=(doc_id,))
