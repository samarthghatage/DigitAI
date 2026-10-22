import numpy as np
from src.inference import predict

def test_inference_output_contract():
    result=predict("lenet",tensor=np.zeros((1,28,28,1),dtype=np.float32))
    assert 0 <= result["digit"] <= 9
    assert len(result["probabilities"]) == 10 and abs(sum(result["probabilities"])-1) < 1e-5
    assert result["tensor"].shape == (1,28,28,1)
