from edcr.condition import Condition
from edcr.edcr_error_detector import EdcrDetRuleLearnErrorDetector, EdcrRatioDetRuleLearnErrorDetector

import pandas as pd

def make_feature_lambda(feature_name):
    return lambda x: x[feature_name] == 1
    
data = pd.read_csv("data/fake_2.csv")
feature_columns = [column for column in data.columns if column != "pred" and column != "target"]
conditions = [
   Condition(column, make_feature_lambda(f"{column}"))
   for column in feature_columns
]

detector = EdcrDetRuleLearnErrorDetector(epsilon=0.35)
detector.train(
    data=data.to_dict("records"), 
    pred=data["pred"].tolist(), 
    labels=data["target"].tolist(), 
    conditions=conditions
)

for rule in detector.rules:
    print(rule)

detector.detect(
    data=data.to_dict("records"),
    pred=data["pred"].tolist()
)