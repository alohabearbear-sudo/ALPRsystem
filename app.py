import os
import gc
import sys
import requests
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import easyocr  
from PIL import Image, ImageOps

# --- 0. Streamlit 網頁基本配置 ---
st.set_page_config(
    page_title="AI智慧車牌辨識系統",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. 設定與路徑 ---
MODEL_URL = "https://github.com/alohabearbear-sudo/Car-Plate-Recognition/releases/download/v1/best.pt"
MODEL_PATH = "best.pt"

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        msg = st.empty()
        msg.warning("⏳ 首次啟動，正在下載 YOLO 車牌偵測模型，請稍候...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        msg.empty()
    return YOLO(MODEL_PATH)

@st.cache_resource
def get_ocr():
    return easyocr.Reader(['en'], gpu=False)

# --- 📐 幾何轉正物理外掛 ---
def correct_plate_rotation(img):
    """
    OpenCV 幾何轉正物理外掛：
    透過最小外接矩形計算傾斜角度，並利用仿射變換將車牌徹底轉正。
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img
            
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[-1]
        
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90
            
        if abs(angle) < 0.5:
            return img
            
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return img

# --- 2. 核心辨識邏輯 ---
def process_recognition(img_np, should_flip=False):
    if img_np is None:
        return None, None, "等待輸入...", "0.00%"
        
    h, w = img_np.shape[:2]
    draw_img = img_np.copy()
    
    plate_crop_res = None
    plate_no_res = "❌ 找不到車牌"
    conf_res = "0.00%"
    
    try:
        model = get_model()
        reader = get_ocr()
        
        results = model.predict(img_np, conf=0.4, verbose=False)
        found_plate = False
        
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                found_plate = True
                x1, y1, x2, y2 = map(int, box)
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                original_max_side = max(x2 - x1, y2 - y1)
                new_w_half = int(original_max_side * 0.6)
                new_h_half = int(original_max_side * 0.25)
                
                xmin, xmax = max(0, cx - new_w_half), min(w, cx + new_w_half)
                ymin, ymax = max(0, cy - new_h_half), min(h, cy + new_h_half)
                
                cv2.rectangle(draw_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 6)
                cv2.putText(draw_img, "License Plate", (xmin, ymin-15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                
                plate_crop = img_np[ymin:ymax, xmin:xmax]
                if plate_crop.size == 0: 
                    continue
                
                # =====================================================================
                # 🔥 右線魔改預處理流水線
                # =====================================================================
                # 步驟 1: 物理幾何角度轉正
                fixed_plate = correct_plate_rotation(plate_crop)
                
                try:
                    # 步驟 2: 轉灰階（注意括號已補死）
                    gray_plate = cv2.cvtColor(fixed_plate, cv2.COLOR_RGB2GRAY)
                    
                    # 步驟 3: CLAHE 局部骨架強化，強行拉開局部黑白對比
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    enhanced_plate = clahe.apply(gray_plate)
                    
                    # 步驟 4: 物理外擴純白邊框（15 像素），解決因切圖太貼邊導致的邊緣失真
                    final_for_ocr = cv2.copyMakeBorder(enhanced_plate, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                except Exception:
                    # 安全防線：發生意外時退回原始灰階並放大
                    gray_plate = cv2.cvtColor(fixed_plate, cv2.COLOR_RGB2GRAY)
                    final_for_ocr = cv2.resize(gray_plate, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

                # 將最終魔改完送進 OCR 的影像保留下來，用於網頁右側欄視覺渲染呈現
                plate_crop_res = final_for_ocr
                
                # --- ⚙️ 關鍵修正點：強制限定只讀取英文大寫、數字和減號 ---
                # 注入解碼黑魔法參數：decoder='greedy', contrast_ths=0.1, adjust_contrast=0.5
                _result = reader.readtext(
                    final_for_ocr, 
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
                    decoder='greedy',
                    contrast_ths=0.1,
                    adjust_contrast=0.5
                )
                
                if _result:
                    words = []
                    scores = []
                    for bbox, text_found, prob in _result:
                        text_upper = text_found.upper()
                        # 進一步二次過濾
                        filtered_text = "".join([c for c in text_upper if c.isalnum() or c == '-'])
                        if filtered_text:
                            words.append(filtered_text)
                            scores.append(prob)
                    
                    # --- ⚙️ 雜訊過濾：只呈現最後一個空白後的字串 ---
                    if words:
                        plate_no_res = words[-1]
                        conf_res = f"{np.mean(scores):.2%}"
        
        if not found_plate:
            plate_no_res = "❌ 找不到車牌"
            conf_res = "0.00%"
            
    except Exception as e:
        plate_no_res = f"❌ 辨識異常: {str(e)}"
    finally:
        gc.collect()
        
    return draw_img, plate_crop_res, plate_no_res, conf_res

# --- 3. 前端 CSS 與 💡 Android 相機權限強制喚醒項 ---
st.markdown("""
<style>
    .stMarkdown h1 { color: #1E88E5; text-align: center; font-weight: bold; }
    .stMarkdown h3 { text-align: center; color: #555; }
    div[data-testid="stStatusWidget"],
    .stSpinner,
    div[data-testid="stNotification"] {
        display: none !important;
    }
</style>

<script>
    const observer = new MutationObserver((mutations) => {
        const inputs = document.querySelectorAll('input[type="file"]');
        inputs.forEach(input => {
            if (!input.hasAttribute('accept')) {
                input.setAttribute('accept', 'image/*');
            }
            if (!input.hasAttribute('capture')) {
                input.setAttribute('capture', 'environment');
            }
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# --- 4. 建立 UI 介面 ---
st.markdown("# 🚗 (Beta 版) 車牌一指辨 by Jimmy Chen")
st.markdown("### 🎯 採用 YOLOv8 + easyOCR 架構 ")
st.markdown("##### 📷 點擊按鈕【拍照或上傳相簿照片】即可開啟手機後置鏡頭")

upload_file = st.file_uploader("👉 只要一個動作，即可輕鬆辨識車牌", type=["jpg", "jpeg", "png"], key="jimmy_unified_uploader")

st.write("---")

# --- 5. 畫面渲染雙欄架構 ---
col_left, col_right = st.columns([3, 2])

if upload_file is not None:
    raw_img = Image.open(upload_file)
    fixed_img = ImageOps.exif_transpose(raw_img).convert('RGB')
    if fixed_img.width > 1024:
        fixed_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    img_np = np.array(fixed_img)
    
    res_draw, res_crop, res_text, res_conf = process_recognition(img_np, should_flip=False)
    
    with col_left:
        st.subheader("1. 定位確認")
        if res_draw is not None:
            st.image(res_draw, use_container_width=True)
            
    with col_right:
        st.subheader("2. 終極魔改（轉正+對比強化+15px白框）")
        if res_crop is not None:
            # 由於經過預處理，此時 res_crop 為單通道灰階圖，指定 channels="GRAY" 完美呈現
            st.image(res_crop, channels="GRAY", use_container_width=True)
        st.subheader("🔢 辨識號碼")
        st.info(f"**{res_text}**")
        st.subheader("信心值")
        st.metric(label="Confidence", value=res_conf)
else:
    gc.collect()
    with col_left:
        st.info("💡 請點擊上方按鈕，選擇【拍照】或從【上傳相簿照片】來啟動車牌自動辨識。")
    with col_right:
        st.text_input("🔢 辨識號碼", value="等待輸入...", disabled=True, key="disabled_output_box")

st.markdown("---")
st.markdown("<center>Developed by Jimmy Chen | 2026 High-Performance Native Version</center>", unsafe_allow_html=True)
