class Condition:
    """
    A class for conditions. Conditions represent rules that can be applied to a dataset. Should return True of False    
    """
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def __call__(self, metadata) -> bool: 
        return self.func(metadata)
    
    def __str__(self):
        return self.name