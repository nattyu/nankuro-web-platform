# config.py
import os

# 画像のパス
IMAGE_DIR = "./images/jpeg"
ANSWER_IMAGE_DIR = "./answers"

# フォント設定
FONT_PATH = "fonts/NotoSansJP-Bold.ttf"
FONT_SIZE = 10

# YOLOの最大検出数
YOLO_MAX_DET = 1000

# YOLOv8モデルのパス
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NUMBER_MODEL_PATH = os.path.join(BASE_DIR, "yolo_models_weights/number_detection/best.pt")
KANJI_MODEL_PATH = os.path.join(BASE_DIR, "yolo_models_weights/kanji_detection/best.pt")
KANJI_BLACK_CELL_MODEL_PATH = os.path.join(BASE_DIR, "yolo_models_weights/kanji_black_cell_detection/best.pt")
SEGMENTATION_MODEL_PATH = os.path.join(BASE_DIR, "yolo_models_weights/segmentation/best.pt")

# 認識モデルの重み
NUMBER_MODEL_WEIGHTS = "./recognition_models_weights/number_model/best_model_250407_231040.pth"
KANJI_MODEL_WEIGHTS = "./recognition_models_weights/kanji_model/best_model_250517_035909.pth"

# 漢字リストのパス
KANJI_CLASSES_PATH = "./models/kanji_classes.txt"

# 出力ディレクトリ
CROPPED_OUTPUT_DIR = "./outputs/crops"
OUTPUT_IMAGE_DIR = "./outputs/images"
CSV_OUTPUT_DIR = "./outputs/csv"
JUKUGO_CSV_OUTPUT_DIR = "./outputs/csv/jukugo"

# YOLOの信頼度閾値
YOLO_CONF_THRESHOLD = 0.5
