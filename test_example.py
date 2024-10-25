import warnings
warnings.filterwarnings("ignore")

from edcr.condition import Condition
from edcr.edcr_error_detector import EdcrDetRuleLearnErrorDetector, EdcrRatioDetRuleLearnErrorDetector

import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def make_feature_lambda(feature_name, desired_value, desired_pred_value=None):
    if desired_pred_value == None: 
        return lambda x, pred: x[feature_name] == desired_value
    return lambda x, pred: x[feature_name] == desired_value and pred == desired_pred_value

def make_model_lambda(data, errors, model):
    print(f"Training {model}")
    model = model()
    model.fit(data, errors)
    return lambda x, pred: 1 if model.predict(pd.DataFrame([x]))[0] else 0

TEST_METRICS_FILE="model_test_metrics_2.json"
VAL_METRICS_FILE="model_val_metrics_2.json"
TRAIN_METRICS_FILE="model_train_metrics_2.json"

TEST_OUTPUT_FILE="model_test_outputs_2.csv"
VAL_OUTPUT_FILE = "model_val_outputs_2.csv"
TRAIN_OUTPUT_FILE = "model_train_outputs_2.csv"

feature_columns = []
feature_columns += ["Male", "Female",
"Age_25_29",
"Age_30_34","Age_35_39","Age_40_44","Age_45_49","Age_50_54","Age_55_59","Age_60_64","Age_65_69","Age_70_74","Age_75_79","Age_80_84","Age_85_89",
"Age_90_94",
"Age_95_99",
"Age_NA",
"ACUSON",
"Agfa","Fuji","GE","Canon","Carestream","Camerad","Kodak","Konica","Riverain","Philips","Samsung","Siemens","Toshiba"
]

train = pd.read_csv("data/mace_inference_train.csv")
test = pd.read_csv("data/mace_inference_test.csv")
val = pd.read_csv("data/mace_inference_val.csv")
train["error"] = train["pred"] != train["label"]
test["error"] = test["pred"] != test["label"]
val["error"] = val["pred"] != val["label"]

models = [
    ("MLPClassifier", MLPClassifier),
    ("RandomForestClassifier", RandomForestClassifier),
    ("ExtraTreesClassifier", ExtraTreesClassifier),
    ("BaggingClassifier", BaggingClassifier),
    ("GradientBoostingClassifier", GradientBoostingClassifier),
    ("AdaBoostClassifier", AdaBoostClassifier),
    ("SVC", SVC),
    ("KNeighborsClassifier", KNeighborsClassifier)
]
model_names = [model[0] for model in models]
model_lambdas = [(name, make_model_lambda(train[feature_columns], train["error"], model)) for name, model in models]

def process_dataset(dataset):
    dataset["error"] = dataset["pred"] != dataset["label"]
    for name, model_lambda in model_lambdas:
        dataset[name] = dataset.apply(lambda x: model_lambda(x[feature_columns].to_dict(), x["pred"]), axis=1)  
    return dataset

train = process_dataset(train)
test = process_dataset(test)
val = process_dataset(val)

train.to_json("testus.json", orient="records")

# I have to do this here, dont move it
feature_columns += model_names
conditions= []
conditions += [Condition(f"{column}_is_0__and_pred_1", make_feature_lambda(column, 0, 1)) for column in feature_columns] 
conditions += [Condition(f"{column}_is_1__and_pred_1", make_feature_lambda(column, 1, 1)) for column in feature_columns] 

test_results = []
val_results = []
train_results = []
model_test_predictions = pd.DataFrame()
model_val_predictions = pd.DataFrame()
model_train_predictions = pd.DataFrame()

def evaluate_detector(detector, dataset, epsilon):
    error_detections = detector.detect(
        data=dataset[feature_columns].to_dict("records"),
        pred=dataset["pred"].tolist()
    )

    actual_preds = dataset["pred"].tolist()

    corrected_preds = [2 if error_detections[index] else actual_preds[index] for index in range(len(error_detections))]
    ground_truth = dataset["label"].tolist()

    combined = filter(lambda x : x[0] != 2, [(corrected_preds[index], ground_truth[index]) for index in range(len(error_detections))])
    corrected_preds, ground_truth = zip(*combined)

    detected = classification_report(ground_truth, corrected_preds, labels=[0, 1, 2], output_dict=True)
    
    number_of_true_positives = sum([1 if corrected_preds[index] == 1 and ground_truth[index] == 1 else 0 for index in range(len(error_detections))])
    number_of_false_positives = sum([1 if corrected_preds[index] == 1 and ground_truth[index] == 0 else 0 for index in range(len(error_detections))])
    number_of_true_negatives = sum([1 if corrected_preds[index] != 1 and ground_truth[index] == 0 else 0 for index in range(len(error_detections))])
    number_of_false_negatives = sum([1 if corrected_preds[index] != 1 and ground_truth[index] == 1 else 0 for index in range(len(error_detections))])

    return {
        "epsilon": epsilon, 
        "accuracy": detected["accuracy"],
        "precision_0": detected["0"]["precision"],
        "precision_1": detected["1"]["precision"],
        "recall_0": detected["0"]["recall"],
        "recall_1": detected["1"]["recall"],
        "f1-score_0": detected["0"]["f1-score"],
        "f1-score_1": detected["1"]["f1-score"],
        "number_of_rules": len(detector.rules),
        "rules": [str(rule) for rule in detector.rules],
        "removed_rows": corrected_preds.count(2),
        "number_of_true_positives": number_of_true_positives,
        "number_of_false_positives": number_of_false_positives,
        "number_of_true_negatives": number_of_true_negatives,
        "number_of_false_negatives": number_of_false_negatives
    }, corrected_preds


detector = EdcrRatioDetRuleLearnErrorDetector()
detector.train(
    data=train[feature_columns].to_dict("records"), 
    pred=train["pred"].tolist(), 
    labels=train["label"].tolist(), 
    conditions=conditions
)

results, preds = evaluate_detector(detector, val, "ratio")
val_results.append(results)
model_val_predictions["ratio"] = preds
results, preds = evaluate_detector(detector, test, "ratio")
test_results.append(results)
model_test_predictions["ratio"] = preds
results, preds = evaluate_detector(detector, train, "ratio")
train_results.append(results)
model_train_predictions["ratio"] = preds

epsilon = 0
while epsilon <= 1:
    print("Learning", epsilon)

    detector = EdcrDetRuleLearnErrorDetector(epsilon=epsilon)
    detector.train(
        data=train[feature_columns].to_dict("records"), 
        pred=train["pred"].tolist(), 
        labels=train["label"].tolist(), 
        conditions=conditions
    )

    epsilon_str = f"epsilon_{epsilon}"
    results, preds = evaluate_detector(detector, val, epsilon)
    val_results.append(results)
    model_val_predictions[epsilon_str] = preds
    results, preds = evaluate_detector(detector, test, epsilon)
    test_results.append(results)
    model_test_predictions[epsilon_str] = preds
    results, preds = evaluate_detector(detector, train, epsilon)
    train_results.append(results)
    model_train_predictions[epsilon_str] = preds

    pd.DataFrame(test_results).to_json(TEST_METRICS_FILE, orient="records")
    model_test_predictions.to_csv(TEST_OUTPUT_FILE)

    pd.DataFrame(val_results).to_json(VAL_METRICS_FILE, orient="records")
    model_val_predictions.to_csv(VAL_OUTPUT_FILE)
    
    pd.DataFrame(train_results).to_json(TRAIN_METRICS_FILE, orient="records")
    model_train_predictions.to_csv(TRAIN_OUTPUT_FILE)

    epsilon += 0.01
    epsilon = round(epsilon, 3)
    