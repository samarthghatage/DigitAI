from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, MODELS_DIR, ARTIFACTS_DIR = ROOT / "data", ROOT / "models", ROOT / "artifacts"
HISTORY_DB = DATA_DIR / "digitai_history.db"
SEED, IMAGE_SHAPE = 42, (28, 28, 1)
MODEL_NAMES = {"lenet": "Improved LeNet-5", "cnn": "Augmented CNN"}
for _path in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, ROOT / "sample_images"):
    _path.mkdir(exist_ok=True)

def model_key(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    key = {"improved_lenet-5": "lenet", "improved_lenet_5": "lenet", "basic_cnn": "cnn", "augmented_cnn": "cnn"}.get(key, key)
    if key not in MODEL_NAMES: raise ValueError(f"Unknown model {name!r}; choose lenet or cnn")
    return key

def model_path(name: str) -> Path: return MODELS_DIR / f"{model_key(name)}.keras"
