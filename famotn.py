import streamlit as st
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import torch
from demo import load_checkpoints, make_animation

# تحميل النموذج
generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml', checkpoint_path='checkpoints/vox-cpk.pth.tar')

# تحميل الصورة وفيديو الحركة
st.title("First Order Motion Model for Image Animation")
source_image = st.file_uploader("Upload a source image", type=["png", "jpg", "jpeg"])
driving_video = st.file_uploader("Upload a driving video", type=["mp4"])

if source_image and driving_video:
    # قراءة الصورة والفيديو
    source_image = imageio.imread(source_image)
    driving_video = imageio.mimread(driving_video, memtest=False)

    # تحويل الصورة والفيديو إلى تنسيق مناسب
    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

    # إنشاء التحريك
    predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True)

    # عرض النتائج
    st.video(predictions)