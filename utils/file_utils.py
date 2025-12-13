# utils/file_utils.py
import os
from datetime import datetime

def create_dir(directory):
    """
    指定したディレクトリが存在しなければ作成
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_timestamp():
    """
    現在時刻をフォーマットした文字列を返す
    """
    return datetime.now().strftime("%y%m%d_%H%M%S")

def save_image(image, folder, filename):
    """
    画像を指定したフォルダに保存
    """
    create_dir(folder)
    path = os.path.join(folder, filename)
    import cv2
    cv2.imwrite(path, image)
    return path
