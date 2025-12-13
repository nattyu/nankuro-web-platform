# preprocessing/transforms.py
from torchvision import transforms
from PIL import Image

def transform_image(image, size=64):
    """
    入力PIL画像をリサイズ、テンソル化、正規化する前処理
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)), #, interpolation=Image.LANCZOS
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)
