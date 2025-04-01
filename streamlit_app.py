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
    max_size = (640, 640)
    image.thumbnail(max_size)

    # ส่งรูปไปยัง Roboflow API
  with st.spinner("กำลังประมวลผล..."):
        try:
            # แปลงภาพเป็น base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode()

            # ส่งไปยัง API
            response = requests.post(API_URL, files={"file": ("image.jpg", io.BytesIO(buffered.getvalue()), "image/jpeg")})
            st.write(f"Response Status Code: {response.status_code}")
            st.write(f"Response JSON: {response.text}")

            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])

                if predictions:
                    # สร้าง mask เปล่า
                    mask = Image.new("L", image.size, 0)  # L mode สำหรับ grayscale

                    # วาด mask บนพื้นฐานของ points ใน predictions
                    for pred in predictions:
                        points = pred["points"]
                        polygon = [(point["x"], point["y"]) for point in points]
                        draw = ImageDraw.Draw(mask)
                        draw.polygon(polygon, fill=255)  # วาดเส้น polygon ที่เติมสี 255 (ขาว)

                    # Overlay mask บนภาพต้นฉบับ
                    mask = ImageOps.invert(mask)  # ทำให้ mask ชัดขึ้น
                    segmented_image = Image.composite(image, Image.new("RGB", image.size, (0, 255, 0)), mask)

                    st.image(segmented_image, caption="ผลลัพธ์จากโมเดล", use_column_width=True)
                else:
                    st.write("API ไม่พบวัตถุในภาพ หรือไม่ได้ส่ง mask กลับมา")
            else:
                st.write("เกิดข้อผิดพลาดในการประมวลผล:", response.text)
        except Exception as e:
            st.write(f"เกิดข้อผิดพลาด: {e}")
