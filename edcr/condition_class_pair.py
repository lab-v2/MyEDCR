class ConditionClassPair:
    """
    A class that represents a condition-class pair. A condition-class pair is a pair of a condition and a class.
    """
    def __init__(self, name, func, target_class):
        """
        name: str, the name of the condition
        func: function, a function that takes in metadata and prediction and returns a boolean
        """
        self.name = name
        self.func = func
        self.target_class = target_class

    def __call__(self, metadata) -> bool: 
        return self.func(metadata)
    
    def __str__(self):
        return self.name