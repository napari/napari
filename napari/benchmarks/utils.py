class Skiper:
    def __init__(self, func):
        self.func = func

    def __contains__(self, item):
        return self.func(item)
