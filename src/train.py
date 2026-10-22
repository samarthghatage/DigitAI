import argparse,json,time
import tensorflow as tf
from .config import ARTIFACTS_DIR,MODEL_NAMES,SEED,model_key,model_path
from .data import load_data
from .models import build_model

def train(name,epochs=5,batch_size=128):
    key=model_key(name); tf.keras.utils.set_random_seed(SEED); (x,y),(vx,vy),_=load_data(); m=build_model(key); aug=tf.keras.Sequential([tf.keras.layers.RandomRotation(.08),tf.keras.layers.RandomZoom(.08)])
    checkpoint=tf.keras.callbacks.ModelCheckpoint(model_path(key),monitor="val_accuracy",save_best_only=True,mode="max")
    history=m.fit(aug(x,training=True),y,validation_data=(vx,vy),epochs=epochs,batch_size=batch_size,callbacks=[checkpoint,tf.keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)],verbose=2); m.save(model_path(key))
    (ARTIFACTS_DIR/f"{key}_history.json").write_text(json.dumps(history.history,indent=2)); (ARTIFACTS_DIR/f"{key}_metadata.json").write_text(json.dumps({"model":MODEL_NAMES[key],"key":key,"seed":SEED,"epochs_requested":epochs,"input_shape":[1,28,28,1],"created_at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},indent=2)); return m

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",default="all"); p.add_argument("--epochs",type=int,default=5); a=p.parse_args(); keys=MODEL_NAMES if a.model.lower()=="all" else {model_key(a.model):1}; [train(k,a.epochs) for k in keys]
