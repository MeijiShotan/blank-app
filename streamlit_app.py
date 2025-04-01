import streamlit as st
import requests
import io
from PIL import Image

# ตั้งค่า API ของ Roboflow


ROBOFLOW_API_KEY = "9BCXeL5a6Vgvn8eqPSR8"
PROJECT_ID = "anemia_pcm"
MODEL_VERSION = "2"
API_URL = f"https://detect.roboflow.com/{PROJECT_ID}/{MODEL_VERSION}?api_key={ROBOFLOW_API_KEY}"

st.title("Image Segmentation Viewer")
st.write("อัปโหลดรูปภาพเพื่อดูผลลัพธ์ของโมเดล")

uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปที่อัปโหลด", use_column_width=True)
    
    # ส่งรูปไปยัง Roboflow API
    with st.spinner("กำลังประมวลผล..."):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        response = requests.post(API_URL, files={"file": image_bytes.getvalue()})
        
        if response.status_code == 200:
            result = response.json()
            segmented_image_url = result.get("predictions", [{}])[0].get("image_url", "")
            if segmented_image_url:
                st.image(segmented_image_url, caption="ผลลัพธ์จากโมเดล", use_column_width=True)
            else:
                st.write("ไม่พบผลลัพธ์")
        else:
            st.write("เกิดข้อผิดพลาดในการประมวลผล")
