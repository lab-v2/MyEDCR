import warnings
warnings.filterwarnings("ignore")

from edcr.condition import Condition
from edcr.edcr_error_detector import EdcrDetRuleLearnErrorDetector, EdcrRatioDetRuleLearnErrorDetector

import pandas as pd

data = pd.read_csv("data/train.csv")

error_detector = EdcrDetRuleLearnErrorDetector(epsilon=100, target_class=1)    
error_detector.train(
    data = data.to_dict("records"),
    pred = data["pred"].to_list(),
    labels = data["target"].to_list(),
    conditions= [
        Condition("Feature 0 is equal 1", lambda metadata: metadata["feature_0"] == 1),
        Condition("Feature 1 is equal 1", lambda metadata: metadata["feature_1"] == 1),
        Condition("Feature 2 is equal 1", lambda metadata: metadata["feature_2"] == 1),
    ]
)

error_detector.detect(
    data = [
        {"feature_0": 1, "feature_1": 0, "feature_2": 1},
        {"feature_0": 0, "feature_1": 1, "feature_2": 1},
        {"feature_0": 1, "feature_1": 1, "feature_2": 1},
    ],
    pred = [1, 0, 1],
)

for rule in error_detector.rules:
    print(rule)