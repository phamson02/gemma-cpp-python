from typing import List

class GemmaModel:
    def __init__(
        self,
        tokenizer_path: str,
        compressed_weights_path: str,
        model_type: int,
    ) -> None:
        pass

    @property
    def bos_token(self) -> int: ...
    @property
    def eos_token(self) -> int: ...
    def complete(
        self,
        prompt: str,
    ) -> str: ...
    def tokenize(
        self,
        text: str,
        add_bos: bool = True,
    ) -> List[int]: ...
    def detokenize(
        self,
        tokens: List[int],
    ) -> str: ...
