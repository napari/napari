import os


def always_false(x):
    return False


class Skipper:
    def __init__(
        self,
        func_pr=always_false,
        func_ci=always_false,
        func_always=always_false,
    ):
        self.func_pr = func_pr if "PR" in os.environ else always_false
        self.func_ci = func_ci if "CI" in os.environ else always_false
        self.func_always = func_always

    def __contains__(self, item):
        return (
            self.func_pr(item) or self.func_ci(item) or self.func_always(item)
        )
