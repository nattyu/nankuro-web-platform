# models/yolo_models.py
from ultralytics import YOLO
import config as config

def load_yolo_models():
    """
    YOLOv8モデル（数字用・漢字用）のロード
    """
    model_number = YOLO(config.NUMBER_MODEL_PATH)
    model_kanji = YOLO(config.KANJI_MODEL_PATH)
    model_kanji_black_cell = YOLO(config.KANJI_BLACK_CELL_MODEL_PATH)
    return model_number, model_kanji, model_kanji_black_cell
