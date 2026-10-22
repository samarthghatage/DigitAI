import numpy as np
from PIL import Image
from src.models import build_lenet,build_cnn
from src.preprocessing import preprocess_image,predict_tensor

def test_models_output_contract():
    for model in (build_lenet(),build_cnn()): assert model.output_shape==(None,10)
def test_preprocessing_rgba_polarity_and_blank():
    rgba=np.zeros((50,40,4),dtype=np.uint8); rgba[10:40,16:24,:3]=255; rgba[10:40,16:24,3]=255
    out=preprocess_image(Image.fromarray(rgba)); assert out.shape==(1,28,28,1) and out.dtype==np.float32
    dark=np.full((50,40),255,dtype=np.uint8); dark[10:40,16:24]=0; assert preprocess_image(dark).max()>.5
    try: preprocess_image(np.zeros((28,28),dtype=np.uint8)); assert False
    except ValueError: pass
def test_tensor_bypass(): assert predict_tensor(np.zeros((28,28,1))).shape==(1,28,28,1)
