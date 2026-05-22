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
import traceback

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
    # 如果檔案不存在或太小（損壞），重新下載
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        msg = st.empty()
        msg.warning("⏳ 正在下載 YOLO 車牌偵測模型，請稍候...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        msg.empty()
    return YOLO(MODEL_PATH)

@st.cache_resource
def get_ocr():
    return easyocr.Reader(['en'], gpu=False)

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
                
                plate_crop_res = plate_crop
                
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
                resized = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                
                # --- ⚙️ 關鍵修正點：強制限定只讀取英文大寫、數字和減號 ---
                # 加入 allowlist 參數，OCR 引擎會自動 bypass 中文字（如：台灣省）與不規則雜訊
                _result = reader.readtext(resized, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
                
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
        print("=== FULL TRACEBACK ===")
        print(traceback.format_exc())
        print("=== END TRACEBACK ===")
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
    try:
        upload_file.seek(0)
        raw_img = Image.open(upload_file)
        raw_img.load()
    except Exception as e:
        st.error(f"圖片讀取失敗: {e}")
        print("=== IMAGE LOAD ERROR ===")
        print(traceback.format_exc())
        print("=== END ===")
        st.stop()

    try:
        fixed_img = ImageOps.exif_transpose(raw_img).convert('RGB')
    except Exception:
        print("exif_transpose failed, using raw")
        print(traceback.format_exc())
        fixed_img = raw_img.convert('RGB')

    if fixed_img.width > 1024:
        fixed_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    img_np = np.array(fixed_img)
else:
    gc.collect()
    with col_left:
        st.info("💡 請點擊上方按鈕，選擇【拍照】或從【上傳相簿照片】來啟動車牌自動辨識。")
    with col_right:
        st.text_input("🔢 辨識號碼", value="等待輸入...", disabled=True, key="disabled_output_box")

st.markdown("---")
st.markdown("<center>Developed by Jimmy Chen | 2026 High-Performance Native Version</center>", unsafe_allow_html=True)
