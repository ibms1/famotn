import os
import requests
import torch
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from skimage.transform import resize
from skimage import img_as_ubyte
import tempfile

# النموذج والتكوين
MODEL_PATH = "checkpoints/vox-cpk.pth.tar"
CONFIG_PATH = "config/vox-256.yaml"

def download_model(model_url):
    """تحميل النموذج إذا لم يكن موجوداً"""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        st.info("جاري تحميل النموذج... قد يستغرق هذا بضع دقائق.")
        response = requests.get(model_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = st.progress(0)
        
        with open(MODEL_PATH, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    progress = int((downloaded / total_size) * 100)
                    progress_bar.progress(progress)
        st.success("تم تحميل النموذج بنجاح!")
    return MODEL_PATH

def process_image(image):
    """معالجة الصورة وتحويلها إلى التنسيق المطلوب"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # التأكد من أن الصورة RGB
    image = image.convert('RGB')
    
    # تغيير حجم الصورة مع الحفاظ على النسبة
    target_size = (256, 256)
    image = np.array(image)
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = resize(image, (new_h, new_w), anti_aliasing=True)
    
    # إضافة padding للوصول إلى الحجم المطلوب
    pad_h = (target_size[0] - new_h) // 2
    pad_w = (target_size[1] - new_w) // 2
    image = np.pad(image, ((pad_h, target_size[0] - new_h - pad_h),
                          (pad_w, target_size[1] - new_w - pad_w),
                          (0, 0)), mode='constant')
    
    return image

def process_video(video_file):
    """معالجة الفيديو وتحويله إلى إطارات"""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = process_image(frame)
        frames.append(frame)
        
        progress_bar.progress((i + 1) / total_frames)
    
    cap.release()
    os.unlink(tfile.name)
    
    return np.array(frames)

def main():
    st.title("تحريك الصور باستخدام First Order Motion Model")
    st.write("قم بتحميل صورة ثابتة وفيديو مرجعي لتحريك الصورة")
    
    # تحميل النموذج
    model_url = st.text_input("أدخل رابط النموذج (vox-cpk.pth.tar):", 
                             "https://your-model-url.com/vox-cpk.pth.tar")
    
    if st.button("تحميل النموذج"):
        model_path = download_model(model_url)
    
    # تحميل الصور والفيديو
    source_image = st.file_uploader("تحميل الصورة الثابتة", type=['jpg', 'jpeg', 'png'])
    driving_video = st.file_uploader("تحميل الفيديو المرجعي", type=['mp4', 'mov', 'avi'])
    
    if source_image is not None and driving_video is not None:
        try:
            # معالجة الصورة المصدر
            source_img = Image.open(source_image)
            source_img = process_image(source_img)
            st.image(source_img, caption="الصورة المصدر بعد المعالجة", use_column_width=True)
            
            # معالجة الفيديو
            st.write("جاري معالجة الفيديو...")
            driving_frames = process_video(driving_video)
            
            # تحميل النموذج والمعالجة
            if os.path.exists(MODEL_PATH):
                from demo import load_checkpoints, make_animation
                
                generator, kp_detector = load_checkpoints(config_path=CONFIG_PATH,
                                                        checkpoint_path=MODEL_PATH)
                
                st.write("جاري إنشاء الرسوم المتحركة...")
                with st.spinner("يرجى الانتظار..."):
                    predictions = make_animation(source_img, driving_frames,
                                              generator, kp_detector,
                                              relative=True)
                    
                # حفظ وعرض النتيجة
                output_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                imageio.mimsave(output_video.name, 
                              [img_as_ubyte(frame) for frame in predictions],
                              fps=25)
                
                st.video(output_video.name)
                st.success("تم إنشاء الرسوم المتحركة بنجاح!")
                
                # تنظيف الملفات المؤقتة
                os.unlink(output_video.name)
                
        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")
            st.write("يرجى التأكد من صحة الملفات المدخلة والنموذج")

if __name__ == "__main__":
    main()