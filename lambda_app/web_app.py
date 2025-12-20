import base64
import hashlib
import json
import logging
import os
import random
import sys
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from decimal import Decimal

from fastapi import FastAPI, HTTPException, Request # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import FileResponse, JSONResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from pydantic import BaseModel

import boto3 # type: ignore
from botocore.config import Config as BotoConfig # type: ignore
from botocore.exceptions import ClientError # type: ignore


# ============================================================
# Configuration & Logging
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nankuro_web")

# ============================================================
# Path Setup
# ============================================================
# Add project root to sys.path to allow imports from solver_core, utils, etc.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ============================================================
# Environment Setup (Mirrors Lambda)
# ============================================================
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
os.environ.setdefault("ULTRALYTICS_NO_AUTOINSTALL", "True")
os.environ.setdefault("YOLO_VERBOSE", "False")

# ============================================================
# S3 Configuration
# ============================================================
AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-northeast-1"))
S3_BUCKET = os.getenv("S3_BUCKET")  # 必須（App Runnerの環境変数で設定）
S3_PREFIX = os.getenv("S3_PREFIX", "uploads/")

_s3_client = None


def get_s3_client():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client(
            "s3",
            region_name=AWS_REGION,
            config=BotoConfig(signature_version="s3v4"),
        )
    return _s3_client


def s3_download_bytes(bucket: str, key: str) -> bytes:
    s3 = get_s3_client()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()

# ============================================================
# DynamoDB Configuration
# ============================================================
DDB_TABLE = os.getenv("DDB_TABLE")  # App Runner 環境変数で設定

_ddb = None
def get_ddb():
    global _ddb
    if not DDB_TABLE:
        raise RuntimeError("DDB_TABLE env var is not set")
    if _ddb is None:
        _ddb = boto3.resource("dynamodb", region_name=AWS_REGION)
    return _ddb.Table(DDB_TABLE)


def _to_ddb_safe(obj: Any) -> Any:
    """
    DynamoDB put_item 用に、float を Decimal に変換する。
    dict/list を再帰的に処理する。
    """
    if isinstance(obj, float):
        # 文字列経由で Decimal にするのが安全（2進浮動小数の誤差を拾いにくい）
        return Decimal(str(obj))
    if isinstance(obj, int):
        return obj
    if isinstance(obj, str) or obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, list):
        return [_to_ddb_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_ddb_safe(v) for k, v in obj.items()}
    return obj  # その他はそのまま（必要なら追加）


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI()

# NOTE:
# allow_origins=["*"] と allow_credentials=True はブラウザ仕様上NGになりやすいので、
# 検証段階では credentials を False にします（CORS地雷を避ける）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 起動直後に “少なくともここまで来ている” をログに出す（CloudWatchで確認用）
logger.info("web_app.py loaded. PROJECT_ROOT=%s", PROJECT_ROOT)


@app.on_event("startup")
async def on_startup():
    # 起動イベントが走れば、サーバが立ち上がっている可能性が高い
    logger.info("FastAPI startup event fired.")
    logger.info("AWS_REGION=%s, S3_BUCKET=%s, S3_PREFIX=%s", AWS_REGION, S3_BUCKET, S3_PREFIX)


# ============================================================
# Lazy Import Helpers
# ============================================================
def get_solver_core():
    # solver_core は重い可能性があるのでここで遅延 import
    from solver import solve
    return solve


def get_pandas():
    import pandas as pd
    return pd

# ============================================================
# Rate Limit (Phase1: in-memory)
# ============================================================
JST = timezone(timedelta(hours=9))

def get_client_ip(req: Request) -> str:
    # App Runner/ALB 経由の場合は x-forwarded-for が付くことが多い
    xff = req.headers.get("x-forwarded-for") or req.headers.get("X-Forwarded-For")
    if xff:
        # "client, proxy1, proxy2" の先頭がクライアントIP
        return xff.split(",")[0].strip()
    return req.client.host if req.client else "unknown"

class DailyIpLimiter:
    def __init__(self, limit_per_day: int = 5):
        self.limit = limit_per_day
        self.counts: Dict[str, int] = {}

    def _key(self, ip: str) -> str:
        day = datetime.now(JST).strftime("%Y-%m-%d")
        return f"{ip}|{day}"

    def check_and_increment(self, ip: str) -> Dict[str, Any]:
        key = self._key(ip)
        cur = self.counts.get(key, 0)
        if cur >= self.limit:
            return {"allowed": False, "remaining": 0}

        cur += 1
        self.counts[key] = cur
        return {"allowed": True, "remaining": self.limit - cur}

limiter = DailyIpLimiter(limit_per_day=5)


# ============================================================
# OCR Logic
# ============================================================
_MODELS_CACHE = None


def run_ocr(img_bytes: bytes):
    """
    ここは重いので完全遅延ロード。
    import時に torch / ultralytics / cv2 を触らない。
    """
    logger.info("Starting run_ocr...")

    import cv2  # type: ignore
    import numpy as np

    # Lazy load custom modules (project local)
    import detection
    import grid
    from models import yolo_models, recognition
    from utils import segmentation_preprocess

    # 1) Decode Image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")

    # Preprocessing
    img = segmentation_preprocess.preprocess_with_segmentation(img)

    # Resize if too large (メモリ/速度対策)
    MAX_SIZE = 1980
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2) Load Models (cache)
    global _MODELS_CACHE
    if _MODELS_CACHE is None:
        logger.info("Initializing models cache...")
        model_num, model_kanji, model_black = yolo_models.load_yolo_models()
        rec_kanji_model, rec_kanji_classes = recognition.load_kanji_model()
        rec_num_model, rec_num_classes = recognition.load_number_model()

        _MODELS_CACHE = {
            "yolo_num": model_num,
            "yolo_kanji": model_kanji,
            "rec_kanji": (rec_kanji_model, rec_kanji_classes),
            "rec_num": (rec_num_model, rec_num_classes),
        }

    cache = _MODELS_CACHE

    # 3) Predict / Process
    results_num = cache["yolo_num"].predict(img, conf=0.5, verbose=False)
    results_kji = cache["yolo_kanji"].predict(img, conf=0.5, verbose=False)

    names_num = cache["yolo_num"].names
    names_kji = cache["yolo_kanji"].names
    rec_num_m, rec_num_c = cache["rec_num"]
    rec_kan_m, rec_kan_c = cache["rec_kanji"]

    final_detected, _ = detection.process_detections_y1(
        img,
        img,
        results_num,
        results_kji,
        num_model=rec_num_m,
        num_cls=rec_num_c,
        kan_model=rec_kan_m,
        kan_cls=rec_kan_c,
        font=None,
        names_num=names_num,
        names_kji=names_kji,
        profile=False,
        draw=False,
    )

    pd = get_pandas()
    try:
        df_grid, _ = grid.create_grid_with_threshold(final_detected, image=None)
        df_grid = df_grid.fillna("")
    except Exception as e:
        logger.error("Grid construction failed: %s", e, exc_info=True)
        df_grid = pd.DataFrame([[""]])

    return df_grid, {"ocr_engine": "yolo_v8_real"}


def df_to_board(df):
    return df.values.tolist()


def board_json_to_df(board_json):
    pd = get_pandas()
    return pd.DataFrame(board_json)

def limit_mapping_for_free(mapping, puzzle_id: str, k: int = 5):
    """
    mapping: [{'num': 1, 'kanji': '山', 'conf': 0.93}, ...]
    """
    seed_src = puzzle_id.encode()
    seed = int(hashlib.sha256(seed_src).hexdigest(), 16)
    rng = random.Random(seed)

    idx = list(range(len(mapping)))
    rng.shuffle(idx)
    keep = set(idx[:k])

    return [m for i, m in enumerate(mapping) if i in keep]

def put_solve_cache(puzzle_id: str, device_id: str, result: dict, plan: str):
    now = int(datetime.now(JST).timestamp())
    ttl = now + 7 * 24 * 60 * 60  # 7日

    safe_result = _to_ddb_safe(result)

    table = get_ddb()
    table.put_item(
        Item={
            "PK": f"PUZZLE#{puzzle_id}",
            "SK": f"DEVICE#{device_id}",
            "result": safe_result,
            "created_at": now,
            "ttl": ttl,
            "plan": plan,
            "version": "v1",
        }
    )

def get_solve_cache(puzzle_id: str, device_id: str):
    table = get_ddb()
    resp = table.get_item(
        Key={
            "PK": f"PUZZLE#{puzzle_id}",
            "SK": f"DEVICE#{device_id}",
        }
    )
    return resp.get("Item")


# ============================================================
# Pydantic Models
# ============================================================
class SolveRequest(BaseModel):
    board: List[List[Any]]
    puzzle_id: Optional[str] = None
    device_id: Optional[str] = None


class PresignRequest(BaseModel):
    filename: str
    content_type: str = "image/jpeg"


class PresignResponse(BaseModel):
    upload_url: str
    s3_key: str
    expires_in: int


class OCRRequest(BaseModel):
    # 推奨：S3直PUTしたオブジェクトキー
    s3_key: Optional[str] = None
    # 互換用（古いフロントが base64 を送る場合）
    image_data: Optional[str] = None


class HealthResponse(BaseModel):
    ok: bool


# ============================================================
# Health
# ============================================================
@app.get("/health", response_model=HealthResponse)
async def health():
    # ここは絶対に軽くする（重いimport禁止）
    return {"ok": True}


# ============================================================
# Presign
# ============================================================
@app.post("/presign", response_model=PresignResponse)
async def presign(req: PresignRequest):
    if not S3_BUCKET:
        raise HTTPException(status_code=500, detail="S3_BUCKET env var is not set")

    ext = os.path.splitext(req.filename)[1].lower() or ".jpg"
    date_path = datetime.utcnow().strftime("%Y/%m/%d")
    key = f"{S3_PREFIX}{date_path}/{uuid.uuid4().hex}{ext}"

    s3 = get_s3_client()
    try:
        upload_url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": S3_BUCKET,
                "Key": key,
                "ContentType": req.content_type,
            },
            ExpiresIn=600,
        )
        return {"upload_url": upload_url, "s3_key": key, "expires_in": 600}
    except ClientError as e:
        logger.error("Presign error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# API Endpoints
# ============================================================
@app.post("/api/ocr")
async def api_ocr(req: OCRRequest):
    try:
        if req.s3_key:
            if not S3_BUCKET:
                raise HTTPException(status_code=500, detail="S3_BUCKET env var is not set")
            img_bytes = s3_download_bytes(S3_BUCKET, req.s3_key)
        elif req.image_data:
            img_bytes = base64.b64decode(req.image_data)
        else:
            raise HTTPException(status_code=400, detail="Provide either s3_key or image_data")

        board_df, meta = run_ocr(img_bytes)

        puzzle_id = str(uuid.uuid4())

        return {
            "status": "ok",
            "puzzle_id": puzzle_id,
            "board": df_to_board(board_df),
            "meta": meta,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("OCR Error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/solve")
async def api_solve(request: SolveRequest, req: Request):
    try:
        plan = req.headers.get("X-Plan", "free")
        puzzle_id = request.puzzle_id or "unknown"
        device_id = req.headers.get("X-Device-Id") or request.device_id or "unknown"

        df = board_json_to_df(request.board)

        # ===== paid: cache check first =====
        if plan == "paid":
            cached = get_solve_cache(puzzle_id, device_id)
            if cached and "result" in cached:
                return cached["result"]

        # ===== solve =====
        from solver import solve
        raw = solve(df)

        mapping = raw.get("mapping", [])

        # ===== free restrictions =====
        if plan != "paid":
            mapping_limited = limit_mapping_for_free(mapping, puzzle_id, k=5)
            raw["mapping"] = mapping_limited
            raw["meta"] = {
                "limited": True,
                "visible_vars": len(mapping_limited),
                "total_vars": len(mapping),
            }
        else:
            raw["meta"] = {
                "limited": False,
                "visible_vars": len(mapping),
                "total_vars": len(mapping),
            }

        raw["puzzle_id"] = puzzle_id
        raw["status"] = "ok"

        # ===== paid: put cache =====
        if plan == "paid":
            put_solve_cache(puzzle_id, device_id, raw, plan)

        return raw

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Solve Error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))




# ============================================================
# Frontend (Static)
# ============================================================
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
MAIN_HTML = os.path.join(FRONTEND_DIR, "main.html")


@app.get("/")
async def read_root():
    # frontend が無くても起動で落ちないようにする
    if os.path.exists(MAIN_HTML):
        return FileResponse(MAIN_HTML)
    return JSONResponse(
        {
            "status": "ok",
            "message": "frontend not found. API is running.",
            "hint": "Deploy frontend/ directory if you want to serve HTML.",
        }
    )


# StaticFiles はディレクトリが存在しないと例外で落ちるため、安全に mount
if os.path.isdir(FRONTEND_DIR):
    try:
        app.mount(
            "/",
            StaticFiles(directory=FRONTEND_DIR, html=True),
            name="frontend",
        )
        logger.info("Mounted frontend static directory: %s", FRONTEND_DIR)
    except Exception as e:
        # ここで落とすと TCP health check が失敗するので、ログに残して続行
        logger.error("Failed to mount frontend static directory: %s", e, exc_info=True)
else:
    logger.warning("FRONTEND_DIR does not exist: %s (skip mounting static files)", FRONTEND_DIR)
