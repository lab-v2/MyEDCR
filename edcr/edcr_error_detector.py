from typing import List, Dict

from edcr.condition import Condition, ConditionLabelPair
from sklearn.metrics import precision_score, recall_score


def POS(
    conditions: List[Condition],
    data: List[Dict],
    predictions: List[int],
    label: List[int],
    expected_label=1,
) -> int:
    """
    This function counts the number of samples where the conditions are met and the model's prediction is an error
    """
    count = 0
    for index in range(len(data)):
        current_data = data[index]
        current_pred = 1 if predictions[index] == expected_label else 0
        current_label = 1 if label[index] == expected_label else 0

        at_least_one_condition_is_met = any(
            [condition(current_data, current_pred) for condition in conditions]
        )
        prediction_is_wrong = current_pred != current_label

        if at_least_one_condition_is_met and prediction_is_wrong:
            count += 1

    return count


def POS_T(
    conditions: List[Condition],
    data: List[Dict],
    predictions: List[int],
    label: List[int],
    expected_label=1,
) -> int:
    """
    This function counts the number of samples where the conditions are met and the model's prediction are false positives
    """
    count = 0
    for index in range(len(data)):
        current_data = data[index]
        current_pred = 1 if predictions[index] == expected_label else 0
        current_label = 1 if label[index] == expected_label else 0

        at_least_one_condition_is_met = any(
            [condition(current_data, current_pred) for condition in conditions]
        )
        is_false_positive = current_pred == 1 and current_label == 0

        if at_least_one_condition_is_met and is_false_positive and current_label == 1:
            count += 1

    return count


def NEG(
    conditions: List[Condition],
    data: List[Dict],
    predictions: List[int],
    label: List[int],
    expected_label=1,
) -> int:
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
        current_pred = 1 if predictions[index] == expected_label else 0
        current_label = 1 if label[index] == expected_label else 0

        at_least_one_condition_is_met = any(
            [condition(current_data, current_pred) for condition in conditions]
        )
        prediction_is_correct = current_pred == current_label

        if at_least_one_condition_is_met and prediction_is_correct:
            count += 1
    return count


def BOD(
    conditions: List[Condition],
    data: List[Dict],
    predictions: List[int],
    label: List[int],
    expected_label=1,
) -> int:
    """
    Counts the number of examples in T that satisfy any of the conditions in DCy and were predicted as y in Y T
    """
    count = 0

    for index in range(len(data)):
        current_data = data[index]
        current_pred = 1 if predictions[index] == expected_label else 0
        # current_label = label[index]

        at_least_one_condition_is_met = any(
            [condition(current_data, current_pred) for condition in conditions]
        )
        prediction_has_value_of_1 = current_pred == 1

        if at_least_one_condition_is_met and prediction_has_value_of_1:
            count += 1

    return count


def POS_CC(
    conditions_label_pairs: List[ConditionLabelPair],
    data: List[Dict],
    predictions: List[int],
    label: List[int],
):
    pass


def BOD_CC(
    conditions_label_pairs: List[ConditionLabelPair],
    data: List[Dict],
    predictions: List[int],
    label: List[int],
):
    pass


def calculate_P(predictions: List[int], label: List[int], expected_label=1) -> float:
    """
    Calculates the precision of all rows where the label is predicted as True.
    This is a value that is calculated in the EDCR paper.
    """
    predictions = [
        predictions[index] for index in range(len(predictions)) if label[index] == 1
    ]
    label = [label[index] for index in range(len(label)) if label[index] == 1]

    return precision_score(label, predictions)


def calculate_false_positives(predictions: List[int], label: List[int]) -> float:
    """
    Calculates the number of false positives in the data.
    """
    predictions = [
        predictions[index] for index in range(len(predictions)) if label[index] == 1
    ]
    label = [label[index] for index in range(len(label)) if label[index] == 1]

    count = 0
    for index in range(len(predictions)):
        if predictions[index] == 1 and label[index] == 0:
            count += 1
    return count


def calculate_R(predictions: List[int], label: List[int]) -> float:
    """
    Calculates the recall of all rows where the label is predicted as True.
    This is a value that is calculated in the EDCR paper.
    """
    predictions = [
        predictions[index] for index in range(len(predictions)) if label[index] == 1
    ]
    label = [label[index] for index in range(len(label)) if label[index] == 1]

    return recall_score(label, predictions)


def DetRuleLearn(
    conditions: List[Condition],
    data,
    predictions: List[int],
    labels: List[int],
    epsilon=0.1,
) -> List[Condition]:
    """
    This is an implementation of the DetRuleLearn algorithm from the EDCR paper.
    Refer to Algorithm 1 in the README for more information.
    """

    N = labels.count(1)
    P = calculate_P(predictions, labels)
    R = calculate_R(predictions, labels)

    threshold = epsilon * (N * P) / R

    learned_conditions = []
    candidate_conditions = list(
        filter(
            lambda condition: NEG([condition], data, predictions, labels) < threshold,
            conditions,
        )
    )

    for condition in candidate_conditions:
        print(condition, end=" ")

    while len(candidate_conditions) > 0:
        best_condition = max(
            candidate_conditions,
            key=lambda condition: POS(
                learned_conditions.copy() + [condition], data, predictions, labels
            ),
        )
        learned_conditions.append(best_condition)
        candidate_conditions.remove(best_condition)
        candidate_conditions = list(
            filter(
                lambda condition: NEG(
                    learned_conditions + [condition], data, predictions, labels
                )
                < threshold,
                candidate_conditions,
            )
        )
    return learned_conditions

def RatioDetRuleLearn(
    conditions: List[Condition], data, predictions: List[int], labels: List[int]
) -> List[Condition]:
    """
    This is an implementation of the RatioDetRuleLearn algorithm from the EDCR paper.
    Refer to Algorithm 2 in the README for more information.
    """
    learned_conditions = []
    list_of_candidate_learned_conditions = (
        []
    )  # Will contain a list of lists with candidate conditions within them.
    candidate_conditions = conditions.copy()
    index = 0

    while len(candidate_conditions) > 0:
        best_condition = min(
            candidate_conditions,
            key=lambda condition: (
                (
                    BOD(learned_conditions + [condition], data, predictions, labels)
                    - BOD(learned_conditions, data, predictions, labels)
                )
                / (
                    POS_T(learned_conditions + [condition], data, predictions, labels)
                    - POS_T(learned_conditions, data, predictions, labels)
                )
            ),
        )
        learned_conditions.append(best_condition)
        list_of_candidate_learned_conditions.append(learned_conditions)

        candidate_conditions.remove(best_condition)
        candidate_conditions = list(
            filter(
                lambda condition: POS_T(
                    learned_conditions + [condition], data, predictions, labels
                )
                > POS_T(learned_conditions, data, predictions, labels),
                candidate_conditions,
            )
        )

        index += 1

    best_learned_conditions = min(
        list_of_candidate_learned_conditions,
        key=lambda cd: (
            BOD(cd, data, predictions, labels)
            + calculate_false_positives(predictions, labels)
        )
        / (POS_T(cd, data, predictions, labels)),
    )
    return best_learned_conditions


# ==================================================================================================
# EDCR Error detectors.
# ==================================================================================================
class EdcrErrorDetector:
    def correct(self, data: List[Dict], pred: List[int]) -> List[int]:
        pass

    def detect(self, data: List[Dict], pred: List[int]) -> List[int]:
        """
        This function detects errors in the data.

        Args:
        data: List[Dict]: The data.
        pred: List[int]: The predictions of the model.
        """
        return [
            any([condition(data[index]) for condition in self.rules])
            for index in range(len(data))
        ]

class EdcrDetRuleLearnErrorDetector(EdcrErrorDetector):
    def __init__(self, epsilon=0.1):
        self.rules = []
        self.epsilon = epsilon

    def train(
        self,
        data: List[Dict],
        pred: List[int],
        labels: List[int],
        conditions: List[Condition],
    ):
        """
        This function trains the error detector.

        Args:
        data: List[Dict]: The data.
        pred: List[int]: The predictions of the model.
        labels: List[int]: The labels of the data.
        conditions: List[Condition]: The conditions to use.
        """
        self.rules = DetRuleLearn(
            conditions=conditions,
            data=data,
            predictions=pred,
            labels=labels,
            epsilon=self.epsilon,
        )


class EdcrRatioDetRuleLearnErrorDetector(EdcrErrorDetector):
    def __init__(self):
        self.rules = []

    def train(
        self,
        data: List[Dict],
        pred: List[int],
        labels: List[int],
        conditions: List[Condition],
    ):
        """
        This function trains the error detector.

        Args:
        data: List[Dict]: The data.
        pred: List[int]: The predictions of the model.
        labels: List[int]: The labels of the data.
        conditions: List[Condition]: The conditions to use.
        """
        self.rules = RatioDetRuleLearn(
            conditions=conditions,
            data=data,
            predictions=pred,
            labels=labels,
        )