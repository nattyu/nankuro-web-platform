import json
import os
import sys
import hashlib
import random
import base64
import logging
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nankuro_web")

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
    import cv2
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
        img, results_num, results_kji,
        rec_num_m, rec_num_c, rec_kan_m, rec_kan_c,
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

# --- API Endpoints ---

@app.post("/api/ocr")
async def api_ocr(request: Request):
    try:
        # Handle both JSON with base64 and raw body if needed, but sticking to JSON for Web
        # The frontend sends { image_data: "base64..." }
        data = await request.json()
        image_data = data.get("image_data")
        
        if not image_data:
            raise HTTPException(status_code=400, detail="No image_data provided")
            
        img_bytes = base64.b64decode(image_data)
        board_df, meta = run_ocr(img_bytes)
        
        return {
            "status": "ok",
            "board": df_to_board(board_df),
            "meta": meta
        }
    except Exception as e:
        logger.error(f"OCR Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/solve")
async def api_solve(request: SolveRequest):
    try:
        # Mock User Context for Web (Simulate standard user or check headers if Auth implemented)
        # For simplicity in this demo, we assume "web_user" and "paid" (or "free")
        # In a real app, you'd check Authorization header.
        user_id = "web_user"
        plan = "paid" # Unlock full features for the Web App deployment
        
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
