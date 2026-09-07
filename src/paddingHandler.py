class PaddingHandler:

    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen

    def padding(self, input: list[int]) -> list[int]:
        input = input[:self.maxlen]
        input.extend((self.maxlen - len(input))*[0])
        return input

