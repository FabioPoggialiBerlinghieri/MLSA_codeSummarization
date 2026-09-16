class PaddingHandler:

    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen

    def padding(self, input: list[int]) -> list[int]:
        """Truncates or pads the token list to match the specified maxlen using zeros."""
        input = input[:self.maxlen]
        input.extend((self.maxlen - len(input))*[0])
        return input