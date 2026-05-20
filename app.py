import streamlit as st
import cv2
import numpy as np
import PIL.Image as Image
from ultralytics import YOLO
import easyocr
import re

# =====================================================================
# 1. 初始化與快取機制（避免重覆載入耗費算力）
# =====================================================================
@st.cache_resource
def load_yolo_model():
    # 載入你訓練好的黃金權重 best.pt，請確保路徑正確
    return YOLO("best.pt")

@st.cache_resource
def load_ocr_reader():
    # 預載入 EasyOCR 英文模型（台灣車牌皆為英數字組合）
    return easyocr.Reader(['en'], gpu=False)

try:
    model = load_yolo_model()
    reader = load_ocr_reader()
except Exception as e:
    st.error(f"模型載入失敗，請檢查 best.pt 是否在倉庫中。錯誤訊息: {e}")

# =====================================================================
# 2. 核心幾何演算法：大津二值化 + 仿射變換角度精準轉正外掛
# =====================================================================
def correct_plate_rotation(plate_img):
    """
    利用 OpenCV 最小外接矩形強行導正斜向車牌
    """
    try:
        if plate_img is None or plate_img.size == 0:
            return plate_img
            
        # 1. 轉灰階並進行高斯模糊去噪
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. 大津二值化 (Otsu's Thresholding) 逼出車牌輪廓
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 3. 尋找車牌外框輪廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return plate_img # 找不到輪廓則回傳原圖
            
        # 抓出面積最大的輪廓（即車牌主體）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 4. 獲取最小外接矩形 (包含中心點、寬高、傾斜角度)
        rect = cv2.minAreaRect(largest_contour)
        center, size, angle = rect
        
        # 5. OpenCV 4.5+ 新舊版本角度定義相容性修正
        if size[0] < size[1]:
            angle = angle + 90
            
        # 限制合理旋轉範圍，避免正常車牌被轉成垂直（車牌傾斜通常在 45 度內）
        if angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = angle + 90
            
        # 如果傾斜角微乎其微，直接過濾不轉，節省運算效能
        if abs(angle) < 1.0:
            return plate_img
            
        # 6. 進行仿射旋轉矩陣計算 (以矩形中心為軸心旋轉)
        (h, w) = plate_img.shape[:2]
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(plate_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    except Exception as e:
        # 演算法容錯，若噴錯則吐回原圖，確保系統不斷聯
        return plate_img

# =====================================================================
# 3. 台灣車牌正規表達式白名單過濾
# =====================================================================
def clean_and_format_plate(text):
    """
    清洗 OCR 雜訊，並嚴格匹配台灣常見車牌格式 (如 ABC-1234, 1234-AB, AAA-123 等)
    """
    # 轉大寫，並移除所有非英數字的雜訊
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # 台灣標準車牌正規匹配規則集
    patterns = [
        r'^[A-Z]{2,3}\d{3,4}$',  # 新/舊式汽車牌 (例: ABC1234, AB1234)
        r'^\d{3,4}[A-Z]{2}$',    # 舊式車牌反向 (例: 1234AB)
        r'^[A-Z]\d[A-Z]\d{4}$',  # 電動車特殊格式
    ]
    
    for pattern in patterns:
        if re.match(pattern, clean_text):
            # 自動加上台灣車牌招牌的「-」小橫線增加可讀性
            if len(clean_text) == 7 and clean_text[:3].isalpha(): # ABC1234 -> ABC-1234
                return f"{clean_text[:3]}-{clean_text[3:]}"
            elif len(clean_text) == 6 and clean_text[:2].isalpha(): # AB1234 -> AB-1234
                return f"{clean_text[:2]}-{clean_text[2:]}"
            elif len(clean_text) == 6 and clean_text[:4].isdigit(): # 1234AB -> 1234-AB
                return f"{clean_text[:4]}-{clean_text[4:]}"
            return clean_text
            
    return clean_text if len(clean_text) >= 4 else None

# =====================================================================
# 4. Streamlit 前端 UI 介面設計
# =====================================================================
st.set_page_config(page_title="ALPR 智慧車牌辨識系統", layout="centered")
st.title("🚗 智慧車牌辨識系統 (ALPR System)")
st.subheader("搭載 YOLOv8 偵測器 + OpenCV 角度轉正外掛 + EasyOCR")
st.write("---")

# 支援圖片上傳與相機即時拍攝（支援跨平台 Android/iOS 手機鏡頭喚醒）
source_type = st.radio("選擇影像來源：", ("上傳圖片檔案", "開啟相機拍照"))

image_file = None
if source_type == "上傳圖片檔案":
    image_file = st.file_uploader("請上傳車牌照片 (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
else:
    image_file = st.camera_input("請對準車輛車牌拍攝")

# =====================================================================
# 5. 核心推論與辨識流水線 (Pipeline)
# =====================================================================
if image_file is not None:
    # 讀取圖片並轉為 OpenCV 的 BGR 格式
    pil_image = Image.open(image_file)
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    st.image(pil_image, caption="原始輸入影像", use_container_width=True)
    
    with st.spinner("🚀 YOLOv8 正在精準定位車牌位置
