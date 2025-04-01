import streamlit as st
import requests
import io
from PIL import Image

# ตั้งค่า API ของ Roboflow


ROBOFLOW_API_KEY = "9BCXeL5a6Vgvn8eqPSR8"
PROJECT_ID = "anemia_pcm"
MODEL_VERSION = "2025-03-31 11:26pm"
API_URL = "https://outline.roboflow.com/anemia_pcm/2?api_key=9BCXeL5a6Vgvn8eqPSR8"

st.title("Image Segmentation Viewer")
st.write("อัปโหลดรูปภาพเพื่อดูผลลัพธ์ของโมเดล")

uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="รูปที่อัปโหลด", use_column_width=True)
    
    # ลดขนาดภาพให้ไม่เกิน 512x512
    max_size = (512, 512)
    image.thumbnail(max_size)

    # ส่งรูปไปยัง Roboflow API
    with st.spinner("กำลังประมวลผล..."):
        try:
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="๋PNG")
            response = requests.post(API_URL, files={"file": image_bytes.getvalue()})

            # Debugging: แสดงสถานะ HTTP และข้อความจาก API
            st.write(f"Response Status Code: {response.status_code}")
            st.write(f"Response JSON: {response.text}")

            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])

                if predictions:
                    segmented_image_url = predictions[0].get("image_url", "")
                    if segmented_image_url:
                        st.image(segmented_image_url, caption="ผลลัพธ์จากโมเดล", use_column_width=True)
                    else:
                        st.write("API ไม่ได้ส่ง URL ของรูปที่ segment มา")
                else:
                    st.write("API ไม่พบวัตถุในภาพ")
            else:
                st.write("เกิดข้อผิดพลาดในการประมวลผล:", response.text)
        except Exception as e:
            st.write(f"เกิดข้อผิดพลาด: {e}")
