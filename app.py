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

# --- 2. 核心辨識邏輯 (💡 整合第三條線：高階影像後處理 Pipeline) ---
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
                
                # ==========================================
                # 💡 核心第三條線：影像後處理實作三部曲
                # ==========================================
                
                # 1. 轉灰階並做高倍率高階放大（利於邊緣字元細節銳利化）
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
                resized = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                
                # 2. 高級去噪與大津法二值化（消除反光與環境雜訊，逼出極致黑白對比）
                blurred = cv2.GaussianBlur(resized, (3, 3), 0)
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 3. 斜向車牌自動角度校正
                coords = np.column_stack(np.where(thresh == 0))
                if coords.size > 0:
                    angle = cv2.minAreaRect(coords)[-1]
                    if angle < -45:
                        angle = -(90 + angle)
                    else:
                        angle = -angle
                        
                    # 若傾斜角度大於 1 度，動態生成仿射矩陣將文字強行轉正
                    if abs(angle) > 1.0:
                        (h_r, w_r) = thresh.shape[:2]
                        center = (w_r // 2, h_r // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        thresh = cv2.warpAffine(thresh, M, (w_r, h_r), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                
                # 將處理完的「超純淨黑白轉正車牌圖」丟回右側控制台顯示，讓使用者親眼見證作弊流效果
                plate_crop_res = thresh
                
                # 4. EasyOCR 滿血調用 (強行掛載台灣車牌專用英數字與連字號白名單)
                _result = reader.readtext(
                    thresh, 
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-',
                    paragraph=False
                )
                
                # ==========================================
                # 💡 後處理 Pipeline 結束
                # ==========================================
                
                if _result:
                    words = []
                    scores = []
                    for bbox, text_found, prob in _result:
                        text_upper = text_found.upper().strip()
                        filtered_text = "".join([c for c in text_upper if c.isalnum() or c == '-'])
                        if filtered_text:
                            words.append(filtered_text)
                            scores.append(prob)
                    
                    if words:
                        plate_no_res = " ".join(words)
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
    // 💡 針對 Android 的 Chrome/Line 瀏覽器進行 Web 核心修正
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
st.markdown("### 🎯 請協助測試並反饋 ")
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
        st.subheader("2. 水平精準裁切區域")
        if res_crop is not None:
            st.image(res_crop, use_container_width=True)
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
