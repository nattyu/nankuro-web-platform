# utils/drawing_utils.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import config as config

def load_font():
    """
    日本語フォントの読み込み
    """
    return ImageFont.truetype(config.FONT_PATH, config.FONT_SIZE)

def draw_box_with_text(image, x1, y1, x2, y2, text, font):
    """
    OpenCVとPILを組み合わせて、画像上に矩形と日本語テキストを描画する
    """
    # OpenCVで矩形を描画
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # PILに変換してテキスト描画
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    #draw.text((x1, y1 - 10), text, font=font, fill=(0, 0, 255))
    # 画像をOpenCV形式に戻す
    image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
