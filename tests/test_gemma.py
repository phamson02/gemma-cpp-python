from pygemma import Gemma, ModelType, ModelTraining

TOKENIZER_PATH = "../model/tokenizer.spm"
COMPRESSED_WEIGHTS_PATH = "../model/2b-it-mqa.sbs"
MODEL_TYPE = ModelType.Gemma2B
MODEL_TRAINING = ModelTraining.GEMMA_IT


def test_gemma():
    gemma = Gemma(
        tokenizer_path=TOKENIZER_PATH,
        compressed_weights_path=COMPRESSED_WEIGHTS_PATH,
        model_type=MODEL_TYPE,
        model_training=MODEL_TRAINING,
    )

    assert gemma
    assert gemma.model

    text = "Hello world!"

    tokens = gemma.tokenize(text)
    assert tokens[0] == gemma.bos_token
    assert tokens == [2, 4521, 2134, 235341]
    detokenized = gemma.detokenize(tokens)
    assert detokenized == text

    # without BOS
    tokens_without_bos = gemma.tokenize(text, add_bos=False)
    assert tokens_without_bos[0] != gemma.bos_token
    assert tokens_without_bos == [4521, 2134, 235341]

    text = "2 +"

    generated = gemma(text)

    assert generated
