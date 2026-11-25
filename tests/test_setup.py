def test_imports():
    import torch, transformers, peft, librosa, mlflow, fastapi  # noqa: F401


def test_wer():
    import jiwer
    assert jiwer.wer("góðan daginn", "góðan daginn") == 0.0
    assert jiwer.wer("góðan daginn", "góðan dag") == 0.5