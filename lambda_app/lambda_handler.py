import json
import os
import sys
import hashlib
import random
import base64

# --- AWS Lambda / Docker Read-Only Fix ---
# Libraries like Ultralytics (YOLO) and Matplotlib try to write to HOME or caching dirs.
os.environ['HOME'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ['YOLO_CONFIG_DIR'] = '/tmp'
# Disable YOLO auto-downloads/updates
os.environ['ULTRALYTICS_NO_AUTOINSTALL'] = 'True'
os.environ['YOLO_VERBOSE'] = 'False'

def log(msg):
    print(msg)
    sys.stdout.flush()


# AWS Lambda環境ではルートに展開される想定だが、
# ローカル検証やディレクトリ構成によってはパス調整が必要
# ここでは、lambda_appの親ディレクトリ(プロジェクトルート)をパスに追加して
# solver_core をインポート可能にする
try:
    # solver_core imports will be done lazily
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
except ImportError:
    pass

# --- Lazy Import Wrappers ---
def get_solver_core():
    from solver_core.solve_nankuro import solve_nankuro
    return solve_nankuro

def get_pandas():
    import pandas as pd
    return pd

# --- Business Logic / Restrictions ---

def select_visible_cells(solutions, user_id, puzzle_id, k=5):
    """
    user_idとpuzzle_idに基づいて決定論的にk個のセルを選択する。
    """
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
    """
    課金ユーザー(paid)以外は conf フィールドを削除する
    """
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
    """
    プランに基づく制限を適用するメイン関数
    """
    # 1. 信頼度 (conf) の制御
    result = hide_conf_for_free(result, plan)
    
    # 課金ユーザーなら全表示で終了
    if plan == "paid":
        return result

    # 2. 表示数制限 (Freeは5マス固定)
    if "solutions" in result and result.get("status") == "ok":
        sols = result["solutions"]
        pid = result.get("puzzle_id", "unknown")
        
        limited_sols = select_visible_cells(sols, user_id, pid, k=5)
        
        # 制限適用後の情報をメタデータに記載
        r = dict(result)
        r["solutions"] = limited_sols
        if "meta" not in r:
            r["meta"] = {}
        r["meta"]["limited"] = True
        r["meta"]["visible_cells"] = len(limited_sols)
        r["meta"]["original_count"] = len(sols)
        
        return r
    
    return result

# --- OCR Stub ---
# --- OCR Stub ---
def run_ocr(img_bytes):
    """
    Real OCR implementation using YOLO detection + PyTorch recognition.
    img_bytes: bytes (decoded from base64)
    Returns: DataFrame (the board)
    """
    log("DEBUG: Starting run_ocr...")
    import cv2
    import numpy as np
    
    # Lazy load our custom modules
    log("DEBUG: Importing custom modules (detection, grid, models)...")
    import detection
    import grid
    from models import yolo_models, recognition
    log("DEBUG: Custom modules imported.")
    
    # 1. Decode Image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    
    log(f"DEBUG: Image decoded. Shape: {img.shape}")

    # --- Segmentation Preprocessing ---
    # Crop to the puzzle area using YOLO segmentation
    log("DEBUG: Running segmentation preprocessing...")
    from utils import segmentation_preprocess
    img = segmentation_preprocess.preprocess_with_segmentation(img)
    log(f"DEBUG: Segmentation done. New shape: {img.shape}")

    # --- Optimization: Resize if too large ---
    # Reduces YOLO and Recognition overhead.
    MAX_SIZE = 800
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        log(f"DEBUG: Resizing image from {w}x{h} to {new_w}x{new_h} (scale={scale:.2f})")
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        log(f"DEBUG: Image size {w}x{h} is within limit.")

    # 2. Load Models
    log("DEBUG: Loading models...")
    
    global _MODELS_CACHE
    if '_MODELS_CACHE' not in globals():
        log("DEBUG: Initializing models cache...")
        model_num, model_kanji, model_black = yolo_models.load_yolo_models()
        log("DEBUG: YOLO models loaded.")
        
        # Load recognition models
        rec_kanji_model, rec_kanji_classes = recognition.load_kanji_model()
        rec_num_model, rec_num_classes = recognition.load_number_model()
        log("DEBUG: Recognition models loaded.")
        
        _MODELS_CACHE = {
            'yolo_num': model_num,
            'yolo_kanji': model_kanji,
            'rec_kanji': (rec_kanji_model, rec_kanji_classes),
            'rec_num': (rec_num_model, rec_num_classes)
        }
    else:
        log("DEBUG: Using cached models.")
        
    cache = _MODELS_CACHE
    
    # 3. Predict with YOLO
    log("DEBUG: Running YOLO inference...")
    results_num = cache['yolo_num'].predict(img, conf=0.5, verbose=False)
    results_kji = cache['yolo_kanji'].predict(img, conf=0.5, verbose=False)
    log("DEBUG: YOLO inference done.")
    
    names_num = cache['yolo_num'].names
    names_kji = cache['yolo_kanji'].names
    
    rec_num_m, rec_num_c = cache['rec_num']
    rec_kan_m, rec_kan_c = cache['rec_kanji']
    
    log("DEBUG: Processing detections...")
    # 4. Process Detections (Filter, Recognition, Merge)
    final_detected, _ = detection.process_detections_y1(
        img,
        results_num,
        results_kji,
        rec_num_m,
        rec_num_c,
        rec_kan_m,
        rec_kan_c,
        font=None,
        names_num=names_num,
        names_kji=names_kji,
        profile=False,
        draw=False
    )
    log("DEBUG: Detections processed.")
    
    # 5. Construct Grid
    log("DEBUG: Constructing grid...")
    pd = get_pandas()
    try:
        df_grid, _ = grid.create_grid_with_threshold(final_detected, image=None)
        # Fill NaN with empty string to be JSON serializable
        df_grid = df_grid.fillna("")
        log("DEBUG: Grid constructed successfully.")
    except Exception as e:
        log(f"DEBUG: Grid construction failed: {e}")
        # Fallback to single cell if grid fails
        df_grid = pd.DataFrame([[""]])
        
    return df_grid, {"ocr_engine": "yolo_v8_real"}

def df_to_board(df):
    return df.values.tolist()

def board_json_to_df(board_json):
    pd = get_pandas()
    return pd.DataFrame(board_json)

# --- Lambda Handler ---

def lambda_handler(event, context):
    """
    AWS Lambda Entry Point
    """
    # Just to confirm function started
    print("Function Starting (Pre-Imports)...")

    print("Event received:", json.dumps(event, ensure_ascii=False)[:200] + "...") # Log intro

    # 1. Parse Path and Method
    # HTTP API Payload 2.0 or REST API
    path = event.get("rawPath", "")
    if not path: # Fallback for REST API
        path = event.get("path", "")
        
    http_context = event.get("requestContext", {}).get("http", {})
    method = http_context.get("method", "")
    if not method: # Fallback for REST API
        method = event.get("httpMethod", "GET")

    # 2. Extract User Context from Authorizer
    authorizer = event.get("requestContext", {}).get("authorizer", {})
    
    # JWT Authorizer等の場合、claimsの中にある場合も
    jwt = authorizer.get("jwt", {})
    claims = jwt.get("claims", {})
    
    user_data = authorizer.get("lambda", {}) # Lambda Authorizer output often in 'lambda' key for HTTP API
    if not user_data:
        user_data = authorizer # REST API often puts it directly
    
    user = authorizer.get("user", {})
    if not user and claims:
        user = {"id": claims.get("sub"), "plan": claims.get("custom:plan", "free")}
        
    user_id = user.get("id", "anonymous")
    plan = user.get("plan", "free")
    
    print(f"User Context: id={user_id}, plan={plan}")

    resp_body = {}
    status_code = 200

    try:
        if path == "/api/ocr" and method == "POST":
            # Lazy load pandas here
            log("DEBUG: Importing pandas for OCR...")
            get_pandas() 
            log("DEBUG: Pandas imported.")

            body_str = event.get("body", "{}")
            is_base64 = event.get("isBase64Encoded", False)
            
            # Helper to parse JSON body VS raw
            # main.html sends JSON: { "image_data": "...", "filename": "..." }
            # But the content might be base64 encoded by Gateway if binary media type set?
            # Or just a JSON string.
            
            # Attempt to parse body as JSON first
            json_body = None
            try:
                # If body is base64 encoded, decode first?
                # HTTP API typically sends text body for application/json
                decoded_body_str = body_str
                if is_base64:
                    decoded_body_str = base64.b64decode(body_str).decode('utf-8')
                
                json_body = json.loads(decoded_body_str)
            except Exception:
                json_body = None
            
            img_bytes = None
            if json_body and "image_data" in json_body:
                # It's our JSON format from main.html
                b64_img = json_body["image_data"]
                img_bytes = base64.b64decode(b64_img)
            else:
                # Fallback: Assume raw body is the image (or base64 of image)
                if is_base64:
                    img_bytes = base64.b64decode(body_str)
                else:
                    img_bytes = body_str.encode()

            log(f"DEBUG: Calling run_ocr. Image bytes length: {len(img_bytes) if img_bytes else 0}")
            board_df, meta = run_ocr(img_bytes)
            log("DEBUG: run_ocr completed.")
            
            resp_body = {
                "status": "ok",
                "board": df_to_board(board_df),
                "meta": meta
            }

        elif path == "/api/solve" and method == "POST":
            # Lazy load solver
            print("Importing solver_core...")
            solve_nankuro = get_solver_core()

            body_str = event.get("body", "{}")
            
            # Same body parsing logic could apply, but solve API expects JSON board
            decoded_body_str = body_str
            if event.get("isBase64Encoded", False):
                 decoded_body_str = base64.b64decode(body_str).decode('utf-8')
                 
            body = json.loads(decoded_body_str)
            
            board_data = body.get("board")
            if not board_data:
                raise ValueError("No board data provided")

            board_df = board_json_to_df(board_data)
            
            # Solve
            raw_result = solve_nankuro(board_df, user_id)
            
            # Apply Restrictions
            resp_body = apply_plan_restrictions(raw_result, plan, user_id)

        else:
            status_code = 404
            resp_body = {"status": "error", "message": "Not Found", "path": path}

    except Exception as e:
        status_code = 500
        resp_body = {"status": "error", "message": str(e)}
        import traceback
        traceback.print_exc()

    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(resp_body, ensure_ascii=False)
    }
