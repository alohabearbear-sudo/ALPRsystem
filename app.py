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
    return YOLO("best.pt")

@st.cache_resource
def load_ocr_reader():
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
            return plate_img
            
        # 抓出面積最大的輪廓（即車牌主體）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 4. 獲取最小外接矩形 (包含中心點、寬高、傾斜角度)
        rect = cv2.minAreaRect(largest_contour)
        center, size, angle = rect
        
        # 5. OpenCV 4.5+ 新舊版本角度定義相容性修正
        if size[0] < size[1]:
            angle = angle + 90
            
        # 限制合理旋轉範圍，避免正常車牌被轉成垂直
        if angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = angle + 90
            
        if abs(angle) < 1.0:
            return plate_img
            
        # 6. 進行仿射旋轉矩陣計算
        (h, w) = plate_img.shape[:2]
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(plate_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    except Exception as e:
        return plate_img

# =====================================================================
# 3. 台灣車牌正規表達式白名單過濾
# =====================================================================
def clean_and_format_plate(text):
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    patterns = [
        r'^[A-Z]{2,3}\d{3,4}$',  
        r'^\d{3,4}[A-Z]{2}$',    
        r'^[A-Z]\d[A-Z]\d{4}$',  
    ]
    
    for pattern in patterns:
        if re.match(pattern, clean_text):
            if len(clean_text) == 7 and clean_text[:3].isalpha(): 
                return f"{clean_text[:3]}-{clean_text[3:]}"
            elif len(clean_text) == 6 and clean_text[:2].isalpha(): 
                return f"{clean_text[:2]}-{clean_text[2:]}"
            elif len(clean_text) == 6 and clean_text[:4].isdigit(): 
                return f"{clean_text[:4]}-{clean_text[4:]}"
            return clean_text
            
    return clean_text if len(clean_text) >= 4 else None

# =====================================================================
# 4. Streamlit 前端 UI 介面設計（還原成最純淨的介面）
# =====================================================================
st.set_page_config(page_title="ALPR 智慧車牌辨識系統", layout="centered")
st.title("🚗 智慧車牌辨識系統 (ALPR System)")
st.subheader("搭載 YOLOv8 偵測器 + OpenCV 角度轉正外掛 + EasyOCR")
st.write("---")

# 這裡移除了 Radio Buttons，直接恢復你原本並存或預設的上傳元件
image_file = st.file_uploader("請上傳車牌照片 (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

# =====================================================================
# 5. 核心推論與辨識流水線 (Pipeline)
# =====================================================================
if image_file is not None:
    pil_image = Image.open(image_file)
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    st.image(pil_image, caption="原始輸入影像", use_container_width=True)
    
    with st.spinner("🚀 YOLOv8 正在精準定位車牌位置..."):
        results = model(frame)
        
    plate_count = 0
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            h_max, w_max = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_max, x2), min(h_max, y2)
            
            cropped_plate = frame[y1:y2, x1:x2]
            if cropped_plate.size == 0:
                continue
                
            plate_count += 1
            st.write(f"### 📍 偵測到第 {plate_count} 張車牌區域：")
            
            # 啟動角度轉正
            fixed_plate = correct_plate_rotation(cropped_plate)
            
            # 對比影像
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB), caption="1. 原始斜向切圖", use_container_width=False)
            with col2:
                st.image(cv2.cvtColor(fixed_plate, cv2.COLOR_BGR2RGB), caption="2. 物理外掛自動轉正", use_container_width=False)
                
            with st.spinner("📝 EasyOCR 正在解碼車牌文字..."):
                ocr_results = reader.readtext(fixed_plate)
                
            raw_text = ""
            final_plate_number = None
            
            if ocr_results:
                raw_text = "".join([res[1] for res in ocr_results])
                final_plate_number = clean_and_format_plate(raw_text)
                
            if final_plate_number:
                st.success(f"🎉 **車牌辨識成功：【 {final_plate_number} 】**")
            else:
                if raw_text.strip():
                    st.warning(f"⚠️ 辨識到疑似雜訊文字：【 {raw_text} 】(未通過台灣車牌格式過濾)")
                else:
                    st.error("❌ 無法清楚辨識車牌文字，請調整
