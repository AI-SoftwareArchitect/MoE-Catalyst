
class Output:
    def __init__(self, _temperature, _output):
        self.temperature = _temperature
        self.output = _output

    def to_dict(self):
        return {
            "temperature": self.temperature,
            "output": self.output
        }

    @staticmethod
    def from_dict(d):
        return Output(d['temperature'], d['output'])