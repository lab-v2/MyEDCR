from edcr.condition import Condition
from edcr.edcr_error_detector import EdcrDetRuleLearnErrorDetector, EdcrDetRatioLearnErrorDetector

import pandas as pd
import random

data = pd.read_csv("data/fake_2.csv")
feature_columns = [column for column in data.columns if column != "pred" and column != "target"]
conditions = []
conditions += [Condition(column, lambda x: x[column] == 1) for column in feature_columns]

detector = EdcrDetRatioLearnErrorDetector()
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