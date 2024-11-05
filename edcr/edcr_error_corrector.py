from typing import List, Dict

from sklearn.metrics import precision_score

from edcr.condition_class_pair import ConditionClassPair


def POS(
    condition_class_pairs: List[ConditionClassPair],
    data: List[Dict],
    predictions: List[int],
    label: List[int],
    target_class,
) -> int:
    """
    This function counts the number of samples where the conditions are met and the model's prediction matches the condition's target class.

    :param condition_class_pairs: List[Condition]: A list of conditions.
    :param data: List[Dict]: The data.
    :param predictions: List[int]: The predictions of the model.
    :param label: List[int]: The labels of the data.
    :param target_class: The expected label.
    """

    count = 0
    for i in range(len(data)):
        current_prediction = 1 if predictions[i] == target_class else 0
        current_label = 1 if label[i] == target_class else 0
        if (
            all(
                [
                    condition_class_pair(data[i])
                    and condition_class_pair.target_class == current_prediction
                    for condition_class_pair in condition_class_pairs
                ]
            )
            and current_label == current_prediction
        ):
            count += 1

    return count


def BOD(
    condition_class_pairs: List[ConditionClassPair],
    data: List[Dict],
    predictions: List[int],
    label: List[int],
    target_class,
) -> int:
    """
    This function counts the number of samples that satisfy the body formed with set of conditions.

    :param condition_class_pairs: List[Condition]: A list of conditions.
    :param data: List[Dict]: The data.
    :param predictions: List[int]: The predictions of the model.
    :param label: List[int]: The labels of the data.
    :param target_class: The expected label.
    """

    count = 0
    for i in range(len(data)):
        current_prediction = 1 if predictions[i] == target_class else 0
        current_label = 1 if label[i] == target_class else 0
        if all(
            [
                condition_class_pair(data[i])
                and condition_class_pair.target_class == current_label
                for condition_class_pair in condition_class_pairs
            ]
        ):
            count += 1

    return count


def calculate_P(predictions: List[int], label: List[int], target_class) -> float:
    """
    Calculates the precision of all rows where the label is predicted as True.
    This is a value that is calculated in the EDCR paper.

    :param predictions: List[int]: The predictions of the model.
    :param label: List[int]: The labels of the data.
    :param expected_label: int: The expected label.
    """

    predictions = [
        1 if predictions[index] == target_class else 0
        for index in range(len(predictions))
    ]
    label = [1 if label[index] == target_class else 0 for index in range(len(label))]

    return precision_score(label, predictions)


def CorrRuleLearn(
    data: List[Dict],
    pred: List[int],
    labels: List[int],
    condition_class_pairs: List[ConditionClassPair],
    target_class,
) -> List[int]:
    learned_condition_class_pairs = []
    candidate_condition_class_pairs = condition_class_pairs.copy()
    P = calculate_P(pred, labels, target_class)

    candidate_condition_class_pairs = sorted(
        filter(
            lambda condition_class_pair: POS(
                [condition_class_pair], data, pred, labels, target_class
            )
            / BOD([condition_class_pair], data, pred, labels, target_class)
            <= P,
            candidate_condition_class_pairs,
        ),
        key=lambda condition_class_pair: POS(
            [condition_class_pair], data, pred, labels, target_class
        )
        / BOD([condition_class_pair], data, pred, labels, target_class),
        reverse=True,
    )

    for condition_class_pair in candidate_condition_class_pairs:
        a = POS(
            learned_condition_class_pairs + [condition_class_pair],
            data,
            pred,
            labels,
            target_class,
        ) / BOD(
            learned_condition_class_pairs + [condition_class_pair],
            data,
            pred,
            labels,
            target_class,
        ) - POS(
            learned_condition_class_pairs, data, pred, labels, target_class
        ) / BOD(
            learned_condition_class_pairs, data, pred, labels, target_class
        )

        temp_candidate_condition_class_pairs = learned_condition_class_pairs.copy()
        temp_candidate_condition_class_pairs.remove(condition_class_pair)

        b = POS(
            temp_candidate_condition_class_pairs, data, pred, labels, target_class
        ) / BOD(
            temp_candidate_condition_class_pairs, data, pred, labels, target_class
        ) - POS(
            candidate_condition_class_pairs, data, pred, labels, target_class
        ) / BOD(
            candidate_condition_class_pairs, data, pred, labels, target_class
        )

        if a >= b:
            learned_condition_class_pairs.append(condition_class_pair)
        else:
            candidate_condition_class_pairs.remove(condition_class_pair)

    if (
        POS(learned_condition_class_pairs, data, pred, labels, target_class)
        / BOD(learned_condition_class_pairs, data, pred, labels, target_class)
        <= P
    ):
        learned_condition_class_pairs = []
    return learned_condition_class_pairs


# ==================================================================================================
# EDCR Error correctors.
# ==================================================================================================
class EdcrErrorCorrector:
    def __init__(self, target_class):
        self.target_class = target_class
        self.condition_class_pairs = []

    def correct(self, data: List[Dict], pred: List[int]) -> List[int]:
        corrected_preds = []
        for index in range(len(data)):
            any_condition_class_pair_is_true = any(
                [
                    condition_class_pair(data[index])
                    and condition_class_pair.target_class == pred[index]
                    for condition_class_pair in self.condition_class_pairs
                ]
            )

            if any_condition_class_pair_is_true:
                corrected_preds.append(self.target_class)
            else:
                corrected_preds.append(pred[index])
        return corrected_preds


class EdcrCorrRuleLearnCorrector:
    def __init__(self, target_class):
        super().__init__(target_class)

    def train(
        self,
        data: List[Dict],
        pred: List[int],
        labels: List[int],
        condition_class_pairs: List[ConditionClassPair],
    ):
        self.condition_class_pairs = CorrRuleLearn(
            data=data,
            pred=pred,
            labels=labels,
            condition_class_pairs=condition_class_pairs,
            target_class=self.target_class,
        )
