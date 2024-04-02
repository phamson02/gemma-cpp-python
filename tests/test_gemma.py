from pygemma import Gemma, ModelType

TOKENIZER_PATH = "../model/tokenizer.spm"
COMPRESSED_WEIGHTS_PATH = "../model/2b-it-sfp.sbs"
MODEL_TYPE = ModelType.Gemma2B


def test_gemma():
    gemma = Gemma(
        tokenizer_path=TOKENIZER_PATH,
        compressed_weights_path=COMPRESSED_WEIGHTS_PATH,
        model_type=MODEL_TYPE,
    )

    assert gemma
    assert gemma.model

    text = "Hello world!"

    tokens = gemma.tokenize(text)  # type: ignore
    assert tokens[0] == gemma.bos_token
    assert tokens == [2, 4521, 2134, 235341]
    detokenized = gemma.detokenize(tokens)  # type: ignore
    assert detokenized == text

    # without BOS
    tokens_without_bos = gemma.tokenize(text, add_bos=False)  # type: ignore
    assert tokens_without_bos[0] != gemma.bos_token
    assert tokens_without_bos == [4521, 2134, 235341]

    generated = gemma(text)  # type: ignore

    assert generated
    assert generated == "Hi there!"
