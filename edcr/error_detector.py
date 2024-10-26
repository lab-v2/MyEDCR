from edcr.detr_rule_learn import DetRuleLearn

class EdcrDetRuleLearnErrorDetector:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.rules = {}

    def train(self, data, pred, labels, conditions):
        unique_labels = set(labels)
        self.rules = dict()

        for label in unique_labels:
            self.rules[label] = DetRuleLearn(
                target_class=label,
                conditions=conditions,
                X=data,
                y=labels,
                y_predictions=pred,
                epsilon=self.epsilon,
            )

    def detect(self, data, pred):
        return [
            any([rule(data[index], pred[index]) for rule in self.rules[pred[index]]])
            for index in range(len(data))
        ]
