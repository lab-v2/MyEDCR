from edcr.condition import Condition
from edcr.edcr_error_detector import EdcrDetRuleLearnErrorDetector

import pandas as pd

data = pd.read_csv("data/fake_1.csv")
feature_columns = [column for column in data.columns if column != "pred" and column != "target"]
conditions = [Condition(column, lambda x: x[column] == 1) for column in feature_columns]
conditions.append(Condition("best rule", lambda x: x["pred"] != x["target"]))

detector = EdcrDetRuleLearnErrorDetector(epsilon=0.1)
detector.train(
    data=data.to_dict("records"), 
    pred=data["pred"].tolist(), 
    labels=data["target"].tolist(), 
    conditions=conditions
)

for rule in detector.rules:
    print(rule)

detector.detect(data.to_dict("records"), data["pred"].tolist())