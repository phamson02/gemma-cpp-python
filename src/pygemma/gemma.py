import os
from typing import Optional

import pygemma.gemma_cpp as gemma_cpp


class Gemma:
    def __init__(
        self,
        *,
        tokenizer_path: str,
        compressed_weights_path: str,
        model_type: str,
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

        if self.model_type not in ["2b-it", "2b-pt", "7b-it", "7b-pt"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.model = gemma_cpp.load_gemma_model(
            self.tokenizer_path.encode("utf-8"),
            self.compressed_weights_path.encode("utf-8"),
            self.model_type.encode("utf-8"),
        )

        assert self.model

    def __call__(
        self,
        prompt: str,
        max_tokens: Optional[int] = 1024,
        temperature: float = 1.0,
    ): ...
