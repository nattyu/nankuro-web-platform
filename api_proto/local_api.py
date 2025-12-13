from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import pandas as pd
import json

# プロジェクトルートをパスに追加して solver_core をインポート可能にする
# このファイルは api_proto/local_api.py なので、親の親ディレクトリではなく親ディレクトリがルート
# c:\DevWorking\nankuro_solver_website\api_proto\local_api.py -> parent is api_proto -> parent is root
# いや、実行時のカレントディレクトリに依存するが、安全のため相対パスで追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solver_core.solve_nankuro import solve_nankuro

app = FastAPI()

class SolveRequest(BaseModel):
    board: list[list[str]] # 2D array of strings
    user_id: str | None = None

@app.post("/api/solve")
async def api_solve(request: SolveRequest):
    """
    Solver API endpoint.
    Receives grid data (2D array), converts to DataFrame, and calls solver logic.
    """
    try:
        # 2D配列をDataFrameに変換
        df = pd.DataFrame(request.board)
        # Solve
        result = solve_nankuro(df, user_id=request.user_id)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ocr")
async def api_ocr():
    """
    OCR API endpoint (Stub).
    """
    return {"status": "ok", "message": "OCR endpoint not implemented yet", "board": []}
