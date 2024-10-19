from typing import List, Dict

from edcr.condition import Condition
from sklearn.metrics import precision_score, recall_score

def POS(conditions: List[Condition], data: List[Dict], predictions: List[int], label: List[int]) -> int:
    """
    This function counts the number of samples where the conditions are met and the model's prediction is an error
    """
    count = 0

    for index in range(len(data)):
        current_data = data[index]
        current_pred = predictions[index]
        current_label = label[index]

        at_least_one_condition_is_met = any([condition(current_data) for condition in conditions])
        prediction_is_wrong = current_pred != current_label

        if at_least_one_condition_is_met and prediction_is_wrong: count += 1
    
    return count

def NEG(conditions: List[Condition], data: List[Dict], predictions: List[int], label: List[int]) -> int: 
    """
    Counts the samples where the conditions are met, but the prediction was correct.

    Args:
    conditions: List[Condition]: A list of conditions.
    data: List[Dict]: The data.
    pred: List[int]: The predictions of the model.
    label: List[int]: The labels of the data.
    """
    count = 0

    for index in range(len(data)):
        current_data = data[index]
        current_pred = predictions[index]
        current_label = label[index]

        at_least_one_condition_is_met = any([condition(current_data) for condition in conditions])
        prediction_is_correct = current_pred == current_label

        if at_least_one_condition_is_met and prediction_is_correct: count += 1
    
    return count

def calculate_P(predictions: List[int], label: List[int]) -> float:
    """
    Calculates the precision of all rows where the label is predicted as True.
    This is a value that is calculated in the EDCR paper.
    """
    predictions = [predictions[index] for index in range(len(predictions)) if label[index] == 1]
    label = [label[index] for index in range(len(label)) if label[index] == 1]

    return precision_score(label, predictions)

def calculate_R(predictions: List[int], label: List[int]) -> float:
    """
    Calculates the recall of all rows where the label is predicted as True.
    This is a value that is calculated in the EDCR paper.
    """
    predictions = [predictions[index] for index in range(len(predictions)) if label[index] == 1]
    label = [label[index] for index in range(len(label)) if label[index] == 1]

    return recall_score(label, predictions)

def RatioDetRuleLearn(self, conditions: List[Condition], labels): pass

def DetRuleLearn(conditions: List[Condition], data, predictions: List[int], labels: List[int], epsilon=0.1) -> List[Condition]: 
    """
    Refer to Algorithm 1.1 in the README for more information.
    """

    N = labels.count(True)
    P = calculate_P(predictions, labels)
    R = calculate_R(predictions, labels)

    threshold = epsilon * (N * P) / R

    learned_conditions = [] 
    DC_star = list(filter(lambda condition: NEG([condition], data, predictions, labels) < threshold, conditions)) 
    while len(DC_star) > 0: 
        c_best = max(DC_star, key=lambda condition: POS(learned_conditions + [condition], data, predictions, labels))
        learned_conditions.append(c_best)
        DC_star.remove(c_best)
        DC_star = list(filter(lambda condition: NEG(learned_conditions + [condition], data, predictions, labels) < threshold, DC_star))
    return learned_conditions

class EdcrDetRuleLearnErrorDetector:
    def __init__(self, epsilon=0.1):
        self.rules = []
        self.epsilon = epsilon

    def train(self, data: List[Dict], pred: List[int], labels: List[int], conditions: List[Condition]):
        self.rules = DetRuleLearn(
            conditions=conditions, 
            data=data, 
            predictions=pred, 
            labels=labels, 
            epsilon=self.epsilon
        )

    def detect(self, data: List[Dict], pred: List[int]) -> List[int]:
        return [any([condition(d) for condition in self.rules]) for d in data]