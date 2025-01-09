# from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks
# from fastapi.responses import FileResponse, JSONResponse
# import os
# import shutil
# from concurrent.futures import ProcessPoolExecutor
# import asyncio
# import uuid
# from model.predict import img_test
# import io
# from tempfile import NamedTemporaryFile


# router = APIRouter(
#     prefix="/animal",
#     tags=["animal"],
# )

# # Chỉ chấp nhận các phần mở rộng file này
# ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# # Tạo Executor cho các tiến trình song song
# executor = ProcessPoolExecutor()

# @router.post("/predict-image/")
# async def process_image(file: UploadFile = File(...)):
#     try:
#         file_ext = os.path.splitext(file.filename)[-1].lower()
#         if file_ext not in ALLOWED_EXTENSIONS:
#             return JSONResponse(content={"info": "Invalid file type."}, status_code=400)

#         # Tạo file tạm để lưu ảnh tải lên
#         with NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
#             temp_path = temp_file.name
#             temp_file.write(await file.read())  # Ghi dữ liệu ảnh vào file tạm

#         object_name, probability = img_test(img_path=temp_path)

#         # Trả về kết quả dự đoán
#         return JSONResponse(content={"message": f"This picture is {object_name}."})

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)
    
#     finally:
#         # Xóa file tạm sau khi xử lý
#         if os.path.exists(temp_path):
#             os.remove(temp_path)



from fastapi import APIRouter, UploadFile, File
from fastapi import Request
from fastapi.responses import JSONResponse
import os
import logging
from concurrent.futures import ProcessPoolExecutor
import asyncio
from tempfile import NamedTemporaryFile
from model.predict import img_test

router = APIRouter(
    prefix="/animal",
    tags=["animal"],
)
# router = APIRouter()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

executor = ProcessPoolExecutor()

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

@router.post("/predict-image/")
async def process_image(file: UploadFile = File(...)):
    temp_path = None  # Khai báo trước để tránh lỗi
    try:
        # Kiểm tra phần mở rộng file
        file_ext = os.path.splitext(file.filename)[-1].lower()
        if not file.filename or file_ext not in ALLOWED_EXTENSIONS:
            return JSONResponse(content={"info": "Invalid or missing file type."}, status_code=400)

        # Tạo file tạm để xử lý
        with NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_path = temp_file.name
            temp_file.write(await file.read())

        # Dự đoán ảnh bằng mô hình
        logging.info(f"Processing file: {file.filename}")
        object_name, probability = await asyncio.get_event_loop().run_in_executor(executor, img_test, temp_path)

        # Phản hồi kết quả
        return JSONResponse(content={
            "info": f"This picture is {object_name}."
        })

    except Exception as e:
        logging.error(f"Error processing file: {file.filename} - {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

