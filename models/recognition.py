import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
import config as config
import os

def load_kanji_classes(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

#from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
def load_kanji_model():
    """
    EfficientNet-B0 ベースの漢字認識モデルを読み込み。
    グレースケール入力かつ多クラス分類に対応。
    """
    # 漢字クラスリストを読み込み
    kanji_classes = load_kanji_classes(config.KANJI_CLASSES_PATH)
    # model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    # FIX: Disable download
    model = shufflenet_v2_x1_0(weights=None)
    
    # 最終層を、漢字クラス数に合わせて修正（ShuffleNet V2 の最終層は model.fc）
    model.fc = nn.Linear(model.fc.in_features, len(kanji_classes))
    
    # グレースケール画像用に、最初の畳み込み層（model.conv1[0]）の入力チャネルを1に変更
    model.conv1[0] = nn.Conv2d(
        in_channels=1,
        out_channels=model.conv1[0].out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    )

    # ── 重みのロード ──
    checkpoint = torch.load(config.KANJI_MODEL_WEIGHTS, map_location=torch.device('cpu'))
    # checkpoint が dict なら 'model_state_dict' キーを使い、それ以外はそのまま
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)

    model.eval()
    return model, kanji_classes

def load_number_model():
    """
    数字認識モデル（ShuffleNet V2）の初期化と重みのロード
    """
    number_classes = [str(i) for i in range(10)]
    # model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    # FIX: Do not download ImageNet weights (access denied/timeout in Lambda). 
    # We load our own weights anyway.
    model = shufflenet_v2_x1_0(weights=None)
    
    # 最終層を、数字クラス数に合わせて修正（ShuffleNet V2 の最終層は model.fc）
    model.fc = nn.Linear(model.fc.in_features, len(number_classes))
    
    # グレースケール画像用に、最初の畳み込み層（model.conv1[0]）の入力チャネルを1に変更
    model.conv1[0] = nn.Conv2d(
        in_channels=1,
        out_channels=model.conv1[0].out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False
    )
    
    checkpoint = torch.load(config.NUMBER_MODEL_WEIGHTS, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, number_classes
