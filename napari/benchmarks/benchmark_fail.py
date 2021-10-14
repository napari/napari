class FailSuite:
    def setup(self):
        print('setup')
        self.values = [1, 2, 3]

    def time_remove(self):
        print('time_remove')
        del self.values[2]
