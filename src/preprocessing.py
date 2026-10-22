from pathlib import Path
import numpy as np
from PIL import Image

def _array(image):
    if isinstance(image, (str, Path)): image = Image.open(image)
    if isinstance(image, Image.Image): image = np.array(image)
    arr = np.asarray(image)
    if arr.ndim == 2: gray = arr.astype("float32")
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        rgb, alpha = arr[..., :3].astype("float32"), arr[..., 3:4].astype("float32") / 255
        gray = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2]) * alpha[...,0]
    elif arr.ndim == 3 and arr.shape[-1] >= 3: gray = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
    else: raise ValueError("Expected grayscale, RGB, or RGBA image")
    if gray.max(initial=0) > 1: gray /= 255.0
    return np.clip(gray,0,1).astype("float32")

def preprocess_image(image, *, return_preview=False):
    gray = _array(image)
    if gray.size == 0 or float(gray.max()) < .04: raise ValueError("The image is blank; draw or upload a digit first.")
    if float(gray.mean()) > .5: gray = 1-gray
    mask = gray > max(.08, float(gray.max())*.12); ys, xs = np.where(mask)
    if len(xs)==0: raise ValueError("No foreground digit was detected.")
    pad=max(1,int(round(max(np.ptp(ys),np.ptp(xs))*.12))); y0,y1=max(0,ys.min()-pad),min(gray.shape[0],ys.max()+pad+1); x0,x1=max(0,xs.min()-pad),min(gray.shape[1],xs.max()+pad+1)
    crop=gray[y0:y1,x0:x1]; scale=20/max(crop.shape); size=(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale)))
    resized=np.array(Image.fromarray((crop*255).astype("uint8")).resize(size,Image.Resampling.LANCZOS),dtype="float32")/255
    canvas=np.zeros((28,28),dtype="float32"); top,left=(28-resized.shape[0])//2,(28-resized.shape[1])//2; canvas[top:top+resized.shape[0],left:left+resized.shape[1]]=resized
    yy,xx=np.indices(canvas.shape); mass=canvas.sum(); cy,cx=((yy*canvas).sum()/mass,(xx*canvas).sum()/mass) if mass else (13.5,13.5); dy,dx=round(13.5-cy),round(13.5-cx)
    shifted=np.zeros_like(canvas); sy0,sy1,sx0,sx1=max(0,dy),min(28,28+dy),max(0,dx),min(28,28+dx); shifted[sy0:sy1,sx0:sx1]=canvas[sy0-dy:sy1-dy,sx0-dx:sx1-dx]
    tensor=shifted[None,...,None].astype("float32"); return (tensor,shifted) if return_preview else tensor

def predict_tensor(tensor):
    arr=np.asarray(tensor,dtype="float32")
    if arr.shape==(28,28): arr=arr[None,...,None]
    elif arr.shape==(28,28,1): arr=arr[None,...]
    if arr.shape!=(1,28,28,1): raise ValueError(f"Expected (1, 28, 28, 1), received {arr.shape}")
    if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
        raise ValueError("Expected finite model input values normalized to [0, 1].")
    return arr
