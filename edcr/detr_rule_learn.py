from typing import List, Dict

from edcr.condition import Condition


def NEG(
    target_class,
    conditions: List[Condition],
    X: List[Dict],
    y: List,
    y_predictions: List,
):
    """
    Calculate the number of train samples that satisfy any of the conditions and are true positive.
    """
    count = 0
    for i in range(len(X)):
        current_X = X[i]
        current_y = y[i]
        current_y_prediction = y_predictions[i]

        satisfies_at_least_one_condition = any([condition(current_X, current_y_prediction) for condition in conditions])
        is_true_positive = current_y == target_class and current_y_prediction == target_class

        if satisfies_at_least_one_condition and is_true_positive:
            count += 1

    return count

def POS(
    target_class,
    conditions: List[Condition],
    X: List[Dict],
    y: List,
    y_predictions: List,
):
    """
    POS calculates the number of samples that satisfy at least one condition and are false positive.
    """
    count = 0
    for i in range(len(X)):
        current_X = X[i]
        current_y = y[i]
        current_y_prediction = y_predictions[i]

        satisfies_at_least_one_condition = any([condition(current_X, current_y_prediction) for condition in conditions])
        is_false_positive = (current_y == target_class and current_y_prediction != target_class)

        if satisfies_at_least_one_condition and is_false_positive:
            count += 1

    return count


def calculate_true_positives(target_class, y: List, y_predictions: List):
    """
    Calculate the number of true positives for a target class

    Args:
    target_class: the class to be detected
    y: List, a list of labels
    y_predictions: List, a list of predictions
    """
    count = 0
    for i in range(len(y)):
        if y[i] == target_class and y_predictions[i] == target_class:
            count += 1
    return count


def calculate_false_positives(target_class, y: List, y_predictions: List):
    """
    Calculate the number of false positives for a target class

    Args:
    target_class: the class to be detected
    y: List, a list of labels
    y_predictions: List, a list of predictions
    """

    count = 0
    for i in range(len(y)):
        if y[i] != target_class and y_predictions[i] == target_class:
            count += 1
    return count


def calculate_true_negatives(target_class, y: List, y_predictions: List):
    """
    Calculate the number of true negatives for a target class

    Args:
    target_class: the class to be detected
    y: List, a list of labels
    y_predictions: List, a list of predictions
    """
    count = 0
    for i in range(len(y)):
        if y[i] != target_class and y_predictions[i] != target_class:
            count += 1
    return count


def calculate_false_negatives(target_class, y: List, y_predictions: List):
    """
    Calculate the number of false negatives for a target class

    Args:
    target_class: the class to be detected
    y: List, a list of labels
    y_predictions: List, a list of predictions
    """
    count = 0
    for i in range(len(y)):
        if y[i] == target_class and y_predictions[i] != target_class:
            count += 1
    return count


def calculate_N(target_class, y: List, y_predictions: List):
    """
    Calculates the number of times the model predicted the target class

    Args:
    target_class: the class to be detected
    y: List, a list of labels
    y_predictions: List, a list of predictions
    """
    count = 0
    for i in range(len(y)):
        if y_predictions[i] == target_class:
            count += 1
    return count


def calculate_P(target_class, y: List, y_predictions: List):
    """
    Calculate the precision for a target class

    Args:
    target_class: the class to be detected
    y: List, a list of labels
    y_predictions: List, a list of predictions
    """

    N = calculate_N(target_class, y, y_predictions)
    TP = calculate_true_positives(target_class, y, y_predictions)
    return TP / N


def calculate_R(target_class, y: List, y_predictions: List):
    """
    Calculate the recall for a target class

    Args:
    target_class: the class to be detected
    y: List, a list of labels
    y_predictions: List, a list of predictions
    """

    N = calculate_N(target_class, y, y_predictions)
    TP = calculate_true_positives(target_class, y, y_predictions)
    FN = calculate_false_negatives(target_class, y, y_predictions)
    return TP / (TP + FN)


def calculate_prior(target_class, y: List, y_predictions: List):
    """
    Calculate the prior for a target class

    Args:
    target_class: the class to be detected
    y: List, a list of labels
    y_predictions: List, a list of predictions
    """

    N = calculate_N(target_class, y, y_predictions)
    return N / len(y)


def DetRuleLearn(
    target_class,
    conditions: List[Condition],
    X: List[Dict],
    y: List,
    y_predictions: List,
    epsilon: float = 0.1,
) -> List[Condition]:
    """
    This is an implementation of Algorithm 1 in the README
    Returns a list of conditions that are rules for error detection

    Args:
    conditions: List[Condition], a list of conditions to be used for error detection\n
    X: List[Dict], a list of metadata\n
    y: List[Dict], a list of labels\n
    y_predictions: List[Dict], a list of predictions\n
    target_class, the class to be detected\n
    epsilon: float, the threshold for error detection\n
    """
    target_class_count = calculate_N(target_class=target_class, y=y, y_predictions=y_predictions)
    precision_of_target_class = calculate_P(target_class=target_class, y=y, y_predictions=y_predictions)
    recall_of_target_class = calculate_R(target_class=target_class, y=y, y_predictions=y_predictions)
    threshold = epsilon * target_class_count * precision_of_target_class / recall_of_target_class

    learned_conditions = []
    candidate_conditions = list(
        filter(
            lambda condition: NEG(
                target_class=target_class,
                conditions=[condition],
                X=X,
                y=y,
                y_predictions=y_predictions,
            ),
            conditions.copy(),
        )
    )

    while len(candidate_conditions) > 0:
        best_condition = max(
            candidate_conditions,
            key=lambda condition: POS(
                target_class=target_class,
                conditions=learned_conditions + [condition],
                X=X,
                y=y,
                y_predictions=y_predictions,
            )
        )

        learned_conditions.append(best_condition)

        candidate_conditions.remove(best_condition)
        candidate_conditions = list(filter(lambda condition: NEG(
            target_class=target_class,
            conditions=learned_conditions + [condition],
            X=X,
            y=y,
            y_predictions=y_predictions
        ) <= threshold, candidate_conditions))
    return learned_conditions