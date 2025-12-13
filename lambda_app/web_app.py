import json
import os
import sys
import hashlib
import random
import base64
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import uuid
from datetime import datetime

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError


# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nankuro_web")

# --- S3 Configuration ---
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


# --- Path Setup ---
# Add project root to sys.path to allow imports from solver_core, utils, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Environment Setup (Mirrors Lambda) ---
os.environ['HOME'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ['YOLO_CONFIG_DIR'] = '/tmp'
os.environ['ULTRALYTICS_NO_AUTOINSTALL'] = 'True'
os.environ['YOLO_VERBOSE'] = 'False'

# --- FastAPI App ---
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lazy Import Helpers ---
def get_solver_core():
    from solver_core.solve_nankuro import solve_nankuro
    return solve_nankuro

def get_pandas():
    import pandas as pd
    return pd

# --- Business Logic (Copied/Adapted from lambda_handler.py) ---

def select_visible_cells(solutions, user_id, puzzle_id, k=5):
    if not user_id:
        user_id = "anonymous"
    if not puzzle_id:
        puzzle_id = "unknown"
        
    seed_src = f"{user_id}:{puzzle_id}".encode()
    seed = int(hashlib.sha256(seed_src).hexdigest(), 16)
    rng = random.Random(seed)
    
    idx = list(range(len(solutions)))
    rng.shuffle(idx)
    chosen_indices = set(idx[:k])
    
    return [solutions[i] for i in chosen_indices]

def hide_conf_for_free(result, plan):
    if plan == "paid":
        return result
    
    r = dict(result)
    if "solutions" in r:
        new_solutions = []
        for s in r["solutions"]:
            d = dict(s)
            d.pop("conf", None)
            new_solutions.append(d)
        r["solutions"] = new_solutions
    return r

def apply_plan_restrictions(result, plan, user_id):
    result = hide_conf_for_free(result, plan)
    if plan == "paid":
        return result

    if "solutions" in result and result.get("status") == "ok":
        sols = result["solutions"]
        pid = result.get("puzzle_id", "unknown")
        limited_sols = select_visible_cells(sols, user_id, pid, k=5)
        
        r = dict(result)
        r["solutions"] = limited_sols
        if "meta" not in r:
            r["meta"] = {}
        r["meta"]["limited"] = True
        r["meta"]["visible_cells"] = len(limited_sols)
        r["meta"]["original_count"] = len(sols)
        return r
    
    return result

# --- OCR Logic ---
_MODELS_CACHE = None

def run_ocr(img_bytes):
    logger.info("Starting run_ocr...")
    import cv2 # type: ignore
    import numpy as np
    
    # Lazy load custom modules
    import detection
    import grid
    from models import yolo_models, recognition
    from utils import segmentation_preprocess

    # 1. Decode Image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    
    # Preprocessing
    img = segmentation_preprocess.preprocess_with_segmentation(img)
    
    # Resize if too large
    MAX_SIZE = 800
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2. Load Models
    global _MODELS_CACHE
    if _MODELS_CACHE is None:
        logger.info("Initializing models cache...")
        model_num, model_kanji, model_black = yolo_models.load_yolo_models()
        rec_kanji_model, rec_kanji_classes = recognition.load_kanji_model()
        rec_num_model, rec_num_classes = recognition.load_number_model()
        
        _MODELS_CACHE = {
            'yolo_num': model_num,
            'yolo_kanji': model_kanji,
            'rec_kanji': (rec_kanji_model, rec_kanji_classes),
            'rec_num': (rec_num_model, rec_num_classes)
        }
    
    cache = _MODELS_CACHE
    
    # 3. Predict / Process
    results_num = cache['yolo_num'].predict(img, conf=0.5, verbose=False)
    results_kji = cache['yolo_kanji'].predict(img, conf=0.5, verbose=False)
    
    names_num = cache['yolo_num'].names
    names_kji = cache['yolo_kanji'].names
    rec_num_m, rec_num_c = cache['rec_num']
    rec_kan_m, rec_kan_c = cache['rec_kanji']
    
    final_detected, _ = detection.process_detections_y1(
        img, img, results_num, results_kji,
        num_model=rec_num_m, num_cls=rec_num_c, kan_model=rec_kan_m, kan_cls=rec_kan_c,
        font=None, names_num=names_num, names_kji=names_kji,
        profile=False, draw=False
    )
    
    pd = get_pandas()
    try:
        df_grid, _ = grid.create_grid_with_threshold(final_detected, image=None)
        df_grid = df_grid.fillna("")
    except Exception as e:
        logger.error(f"Grid construction failed: {e}")
        df_grid = pd.DataFrame([[""]])
        
    return df_grid, {"ocr_engine": "yolo_v8_real"}

def df_to_board(df):
    return df.values.tolist()

def board_json_to_df(board_json):
    pd = get_pandas()
    return pd.DataFrame(board_json)

# --- Pydantic Models for Input ---
class OcrRequest(BaseModel):
    image_data: str # Base64 encoded string
    filename: Optional[str] = None

class SolveRequest(BaseModel):
    board: List[List[Any]]

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

@app.get("/health", response_model=HealthResponse)
async def health():
    return {"ok": True}

@app.post("/presign", response_model=PresignResponse)
async def presign(req: PresignRequest):
    if not S3_BUCKET:
        raise HTTPException(status_code=500, detail="S3_BUCKET env var is not set")

    # 拡張子は filename から推定（無くてもOK）
    ext = os.path.splitext(req.filename)[1].lower() or ".jpg"
    # 例: uploads/2025/12/13/<uuid>.jpg
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
            ExpiresIn=600,  # 10分
        )
        return {"upload_url": upload_url, "s3_key": key, "expires_in": 600}
    except ClientError as e:
        logger.error("Presign error", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# --- API Endpoints ---

@app.post("/api/ocr")
async def api_ocr(req: OCRRequest):
    try:
        # 1) s3_key が来たら S3 から取る（推奨）
        if req.s3_key:
            if not S3_BUCKET:
                raise HTTPException(status_code=500, detail="S3_BUCKET env var is not set")
            img_bytes = s3_download_bytes(S3_BUCKET, req.s3_key)

        # 2) 互換：base64 が来たら decode（ただし大きい画像で413になりやすい）
        elif req.image_data:
            img_bytes = base64.b64decode(req.image_data)

        else:
            raise HTTPException(status_code=400, detail="Provide either s3_key or image_data")

        board_df, meta = run_ocr(img_bytes)

        return {
            "status": "ok",
            "board": df_to_board(board_df),
            "meta": meta
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/solve")
async def api_solve(request: SolveRequest, req: Request):
    try:
        user_id = "web_user"

        # まずはヘッダ優先、無ければ env、最後に paid
        # 例: X-Plan: free / paid
        plan = req.headers.get("X-Plan") or os.getenv("DEFAULT_PLAN", "paid")

        solve_nankuro = get_solver_core()
        board_df = board_json_to_df(request.board)

        raw_result = solve_nankuro(board_df, user_id)
        result = apply_plan_restrictions(raw_result, plan, user_id)

        return result
    except Exception as e:
        logger.error(f"Solve Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Match frontend/main.html logic where it redirects to result_solve.html
# We serve the frontend directory as static files.
# But we need the root URL to serve main.html
@app.get("/")
async def read_root():
    return FileResponse(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'main.html'))

# Mount frontend as static to serve css, js, other htmls
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), '..', 'frontend'), html=True), name="frontend")
