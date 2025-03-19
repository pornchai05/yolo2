import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

model_path = 'yolosegtest.pt'            
model = YOLO(model_path)

st.title("ตรวจจับวัตถุ : ภาพนิ่ง")
img_file = st.file_uploader("เปิดไฟล์ภาพ")

if img_file is not None:    
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #----------------------------------------------
    results = model.predict(source=img, device='cpu', save=False, conf=0.5)
    
    if len(results) > 0:
        for result in results:
            output = result.plot()
        text_obj = 'ตรวจพบ '
    else:
        text_obj = 'ไม่พบวัตถุ'
    #----------------------------------------------
    st.header(text_obj)
    st.image(output, caption='ภาพ Output',channels="BGR")
    
