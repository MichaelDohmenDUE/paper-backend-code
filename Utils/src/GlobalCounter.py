class GlobalCounter:
    def __init__(self):
        self.counter: int = 0

    def set(self, value: int):
        self.counter = value

    def get(self) -> int:
        return self.counter