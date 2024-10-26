class Condition:
    """
    A class for conditions. Conditions represent rules that can be applied to a dataset. Should return True of False    
    """
    def __init__(self, name, func):
        """
        name: str, the name of the condition
        func: function, a function that takes in metadata and prediction and returns a boolean
        """
        self.name = name
        self.func = func

    def __call__(self, metadata, prediction) -> bool: 
        return self.func(metadata, prediction)
    
    def __str__(self):
        return self.name