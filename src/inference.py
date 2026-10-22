import time
import numpy as np
import tensorflow as tf
from .config import model_path
from .models import build_model
from .preprocessing import predict_tensor,preprocess_image

def load_model(name): return tf.keras.models.load_model(model_path(name)) if model_path(name).exists() else None
def predict(name,image=None,tensor=None):
    batch=predict_tensor(tensor) if tensor is not None else preprocess_image(image) if image is not None else (_ for _ in ()).throw(ValueError("Provide image or tensor")); model=load_model(name) or build_model(name); start=time.perf_counter(); probs=np.asarray(model.predict(batch,verbose=0))[0]; elapsed=(time.perf_counter()-start)*1000
    if probs.shape != (10,) or not np.isfinite(probs).all() or probs.sum() <= 0: raise ValueError("Model returned an invalid ten-class output.")
    probs = probs / probs.sum()
    return {"digit":int(np.argmax(probs)),"confidence":float(np.max(probs)),"probabilities":probs.tolist(),"inference_ms":elapsed,"tensor":batch}
