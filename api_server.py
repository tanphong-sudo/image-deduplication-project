# Khởi động API: uvicorn api_server:app --reload
import os
import sys
import json
import logging
import shutil
import tempfile # Dùng để tạo thư mục tạm thời
from pathlib import Path
from typing import Optional, Dict, Any, List

# Thêm thư mục gốc vào PYTHONPATH để có thể import run_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Image Deduplication API")

# ----------------------------------------------------
# A. Cấu hình phục vụ file TĨNH (Static Files)
# ----------------------------------------------------
# Thư mục cache sẽ nằm trong data/processed. Chúng ta cần mount PROJECT_ROOT
app.mount("/static", StaticFiles(directory=PROJECT_ROOT), name="static")

# Cấu hình CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thư mục cache ảnh dành cho web (Nằm trong thư mục dự án để có thể phục vụ)
SERVABLE_IMAGE_CACHE = PROJECT_ROOT / "data/processed" / "session_images"

# Hàm chuyển đổi đường dẫn file local sang URL web hợp lệ
def path_to_url(local_path: str, base_url: str = "http://127.0.0.1:8000") -> str:
    """
    Chuyển đổi đường dẫn file system (dù tương đối hay tuyệt đối) sang URL tĩnh.
    Dùng cho các file đã được copy vào thư mục SERVABLE_IMAGE_CACHE.
    """
    input_path = Path(local_path)
    
    # Ghép nối với thư mục gốc dự án để tạo đường dẫn tuyệt đối chính xác
    full_path_to_file = (PROJECT_ROOT / input_path).resolve()
    
    # Kiểm tra tính hợp lệ và tính relative path
    try:
        relative_path = full_path_to_file.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        logging.warning(f"File path {local_path} is outside project root. Serving placeholder.")
        return "https://placehold.co/150x150/ef4444/ffffff?text=FILE+ERROR"

    return f"{base_url}/static/{relative_path}"

# ----------------------------------------------------
# 1. Định nghĩa cấu trúc dữ liệu đầu vào (Params)
# ----------------------------------------------------
class PipelineParams(BaseModel):
    # Core Params
    extractor: str
    method: str
    euclidean_threshold: Optional[float] = 50.0
    hamming_threshold: Optional[int] = 5
    # Advanced Params
    batch_size: Optional[int] = 16
    k_neighbors: Optional[int] = 50
    pca_dim: Optional[int] = None
    use_gpu: Optional[bool] = False
    index_type: Optional[str] = "flat"
    nlist: Optional[int] = 1024
    simhash_bits: Optional[int] = 64

# ----------------------------------------------------
# 2. Endpoint mới cho File Upload và Xử lý
# ----------------------------------------------------
@app.post("/api/run_pipeline_upload")
async def run_pipeline_upload_endpoint(
    files: List[UploadFile] = File(...), 
    config: str = Form(...) # Cấu hình JSON được gửi dưới dạng chuỗi Form data
):
    
    # 1. PHÂN TÍCH THAM SỐ CẤU HÌNH
    try:
        config_dict = json.loads(config)
        params = PipelineParams(**config_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration JSON: {e}")

    # 2. TẠO THƯ MỤC TẠM THỜI VÀ LƯU FILE
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="dedup_upload_")
    temp_dataset_path = Path(temp_dir_obj.name)
    
    # Đảm bảo thư mục cache tồn tại
    SERVABLE_IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Created temporary directory: {temp_dataset_path}")
    
    try:
        for file in files:
            # Sửa đổi: Đảm bảo tên file an toàn
            safe_filename = file.filename.replace(" ", "_")
            file_path = temp_dataset_path / safe_filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"Saved {len(files)} files to temporary path: {temp_dataset_path}")
        
        # 3. XÂY DỰNG LỆNH CLI
        command = ["python", "run_pipeline.py", "--dataset", str(temp_dataset_path)]

        if params.extractor != 'efficientnet':
            command.extend(["--extractor", params.extractor])
        if params.method != 'faiss':
            command.extend(["--method", params.method])
        if params.method == 'simhash':
            command.extend(["--hamming-threshold", str(params.hamming_threshold)])
        elif params.method == 'faiss' and params.euclidean_threshold != 50.0:
            command.extend(["--threshold", str(params.euclidean_threshold)])
        if params.batch_size != 16:
            command.extend(["--batch-size", str(params.batch_size)])
        if params.k_neighbors != 50:
            command.extend(["--k", str(params.k_neighbors)])
        if params.pca_dim is not None and params.pca_dim > 0:
            command.extend(["--pca-dim", str(params.pca_dim)])
        if params.use_gpu:
            command.append("--use-gpu")
        if params.method == 'faiss':
            if params.index_type != 'flat':
                command.extend(["--index-type", params.index_type])
            if params.index_type == 'ivf' and params.nlist != 1024:
                command.extend(["--nlist", str(params.nlist)])
        if params.method == 'simhash' and params.simhash_bits != 64:
            command.extend(["--simhash-bits", str(params.simhash_bits)])
            
        # 4. CHẠY PIPELINE
        logging.info(f"Executing command: {' '.join(command)}")

        output_json_path = PROJECT_ROOT / "data/processed" / "evaluation_full.json"
        
        if output_json_path.exists():
            os.remove(output_json_path)

        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=False,
            encoding="utf-8"
        )
        
        if result.returncode != 0:
            logging.error(f"Pipeline failed with exit code {result.returncode}")
            error_detail = result.stderr.strip() or result.stdout.strip() or "Unknown error."
            raise HTTPException(status_code=500, detail=f"Pipeline execution failed. Output: {error_detail[:500]}")

        if not output_json_path.exists():
             raise HTTPException(status_code=500, detail="Pipeline finished but did not produce evaluation_full.json.")

        # 5. ĐỌC VÀ CHUẨN BỊ KẾT QUẢ
        with open(output_json_path, 'r') as f:
            metrics = json.load(f)
            
        cluster_file = PROJECT_ROOT / "data/processed" / f"{params.method}_clusters.json"
        
        cluster_data = {"clusters": []}
        if cluster_file.exists():
            with open(cluster_file, 'r') as f:
                full_cluster_data = json.load(f)
                
                if full_cluster_data.get('clusters'):
                    for i, cluster_indices in enumerate(full_cluster_data['clusters']):
                        if cluster_indices:
                            # 5a. Lấy đường dẫn file gốc từ cluster_indices
                            representative_path_temp = full_cluster_data['representatives'].get(str(i))
                            
                            # Lấy TÊN FILE (để tìm trong thư mục TEMP)
                            rep_filename = Path(representative_path_temp).name.replace(" ", "_")
                            
                            # 5b. SAO CHÉP VÀO CACHE TRƯỚC
                            # Đường dẫn đến file ảnh trong thư mục TẠM (để sao chép)
                            src_rep_path = temp_dataset_path / rep_filename 
                            # Đường dẫn đến file ảnh trong thư mục CACHE (để web truy cập)
                            dst_rep_path = SERVABLE_IMAGE_CACHE / rep_filename
                            
                            # Sao chép file đại diện
                            if src_rep_path.exists():
                                shutil.copy(src_rep_path, dst_rep_path)

                            # TẠO URL (Đường dẫn sẽ trỏ đến file trong thư mục CACHE)
                            rep_url = path_to_url(str(dst_rep_path))
                            
                            dup_urls = []
                            for original_dup_path in cluster_indices:
                                if original_dup_path == representative_path_temp:
                                    continue # Bỏ qua ảnh đại diện
                                
                                dup_filename = Path(original_dup_path).name.replace(" ", "_")
                                src_dup_path = temp_dataset_path / dup_filename
                                dst_dup_path = SERVABLE_IMAGE_CACHE / dup_filename
                                
                                if src_dup_path.exists():
                                    shutil.copy(src_dup_path, dst_dup_path)
                                    dup_urls.append(path_to_url(str(dst_dup_path)))
                                    
                            
                            cluster_data['clusters'].append({
                                'id': f'C_{i+1}',
                                'representative': rep_url,
                                'duplicates': dup_urls
                            })

        return {
            "status": "SUCCESS",
            "metrics": metrics,
            "cluster_summary": cluster_data
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("API Error during pipeline execution")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {type(e).__name__}: {str(e)}")
    finally:
        # 6. DỌN DẸP
        temp_dir_obj.cleanup()
        logging.info(f"Cleaned up temporary directory: {temp_dataset_path}")