from pygemma import Gemma

TOKENIZER_PATH = "../model/tokenizer.spm"
COMPRESSED_WEIGHTS_PATH = "../model/2b-it-sfp.sbs"
MODEL_TYPE = "2b-it"


def test_gemma():
    gemma = Gemma(
        tokenizer_path=TOKENIZER_PATH,
        compressed_weights_path=COMPRESSED_WEIGHTS_PATH,
        model_type=MODEL_TYPE,
    )

    assert gemma
    assert gemma.model
