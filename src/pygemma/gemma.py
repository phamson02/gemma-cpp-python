import os
from enum import Enum
from typing import List, Optional

import _pygemma


class ModelType(Enum):
    Gemma2B = 0
    Gemma7B = 1


class Gemma:
    def __init__(
        self,
        *,
        tokenizer_path: str,
        compressed_weights_path: str,
        model_type: ModelType,
    ):
        self.tokenizer_path = tokenizer_path
        self.compressed_weights_path = compressed_weights_path
        self.model_type = model_type

        self.model = None

        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")

        if not os.path.exists(self.compressed_weights_path):
            raise FileNotFoundError(
                f"Compressed weights not found: {self.compressed_weights_path}"
            )

        self.model = _pygemma._GemmaModel(  # type: ignore
            self.tokenizer_path,
            self.compressed_weights_path,
            self.model_type.value,
        )

        assert self.model  # type: ignore

    @property
    def bos_token(self) -> int:
        return self.model.bos_token  # type: ignore

    def __call__(
        self,
        prompt: str,
        max_tokens: Optional[int] = 1024,
        temperature: float = 1.0,
    ) -> str:
        return self.model.complete(prompt)  # type: ignore

    def tokenize(
        self,
        text: str,
        add_bos: bool = True,
    ) -> List[int]:
        return self.model.tokenize(text, add_bos)  # type: ignore

    def detokenize(
        self,
        tokens: List[int],
    ) -> str:
        return self.model.detokenize(tokens)  # type: ignore
