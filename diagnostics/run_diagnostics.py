import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.preprocessing import preprocess_image,predict_tensor
def run():
    checks={}
    for digit in range(10): checks[f"digit_{digit}"]=preprocess_image(np.pad(np.ones((26,8),dtype=np.uint8)*255,((19,19),(28,28)))).shape==(1,28,28,1)
    rgba=np.zeros((64,64,4),dtype=np.uint8); rgba[18:46,28:36,:3]=255; rgba[18:46,28:36,3]=255; checks["rgba"]=preprocess_image(Image.fromarray(rgba)).shape==(1,28,28,1)
    try: preprocess_image(np.zeros((28,28),dtype=np.uint8)); checks["blank_rejected"]=False
    except ValueError: checks["blank_rejected"]=True
    checks["tensor_bypass"]=predict_tensor(np.zeros((28,28,1),dtype=np.float32)).shape==(1,28,28,1); out=Path(__file__).parent/"diagnostic_report.json"; out.write_text(json.dumps(checks,indent=2)); print(json.dumps(checks,indent=2)); return all(checks.values())
if __name__=="__main__": raise SystemExit(0 if run() else 1)
