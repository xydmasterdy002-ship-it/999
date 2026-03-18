import os
import json
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
import torch.nn.functional as F
from deep_translator import GoogleTranslator
from datetime import datetime
import pandas as pd
import gdown
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2

# =========================================================
# Google Drive 模型設定
# =========================================================
MODEL_FILES = {
    "MobileNetV1": {
        "file_id": "1FqpeWNvrSHZk0o0Cka2o_BIXyjPgosMi",
        "path": "MobileNet-V1_imagenet100.pth"
    },
    "GoogLeNet": {
        "file_id": "1SrBz2SQ1VcyEItV-pX6HozJzKkHtJ2AR",
        "path": "googlenet_imagenet100.pth"
    },
    "ResNet101": {
        "file_id": "1RnSGWxh99VG3kb_LN-cx7rp1VWmEFEWs",
        "path": "Resnet-101_imagenet100.pth"
    }
}

LABEL_PATH = "Labels.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translator = GoogleTranslator(source="en", target="zh-TW")

# =========================================================
# 下載模型
# =========================================================
def download_model(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"Downloading {output}..."):
            gdown.download(url, output, quiet=False)

# =========================================================
# Labels
# =========================================================
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    wnid_to_label = json.load(f)

class_names = sorted(wnid_to_label.keys())
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
idx_to_label = {idx: wnid_to_label[cls] for cls, idx in class_to_idx.items()}
num_classes = len(idx_to_label)

# =========================================================
# Transform
# =========================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# =========================================================
# Models
# =========================================================
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True)
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNet101Custom(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.base_model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# =========================================================
# Cached model loader
# =========================================================
@st.cache_resource
def load_model(model_name):
    model_info = MODEL_FILES[model_name]

    download_model(model_info["file_id"], model_info["path"])

    if model_name == "MobileNetV1":
        model = MobileNetV1(num_classes)
    elif model_name == "GoogLeNet":
        model = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet101":
        model = ResNet101Custom(num_classes)

    state_dict = torch.load(model_info["path"], map_location=device)

    if any("module." in k for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = nn.DataParallel(model).to(device)
    model.eval()

    return model

# =========================================================
# UI
# =========================================================
st.title("Multi-Model ImageNet100 Prediction System")

model_name = st.selectbox("Select Model", ["MobileNetV1", "GoogLeNet", "ResNet101"])
model = load_model(model_name)

# =========================================================
# Prediction
# =========================================================
def predict(image):
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x)
        prob = F.softmax(output, dim=1)
        top5_prob, top5_idx = prob.topk(5)

    results = []
    for i in range(5):
        idx = top5_idx[0][i].item()
        label = idx_to_label[idx]
        eng = label.split(",")[0]
        score = top5_prob[0][i].item() * 100
        zh = translator.translate(eng)
        results.append((eng, zh, score))
    return results

# =========================================================
# Upload mode
# =========================================================
upload_mode = st.radio("Upload Mode", ["Single/Multiple Images", "Folder", "Webcam"])
images_list = []

if upload_mode == "Single/Multiple Images":
    uploaded_files = st.file_uploader("Upload", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            images_list.append((f.name, Image.open(f).convert("RGB")))

elif upload_mode == "Folder":
    folder_path = st.text_input("Folder Path")
    if folder_path and os.path.isdir(folder_path):
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg",".jpeg",".png")):
                path = os.path.join(folder_path, fname)
                images_list.append((fname, Image.open(path).convert("RGB")))

elif upload_mode == "Webcam":

    class Cam(VideoProcessorBase):
        def __init__(self):
            self.frames = []

        def recv(self, frame):
            return frame

        def capture(self, frame):
            img = frame.to_ndarray(format="bgr24")
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            self.frames.append(("webcam_" + datetime.now().strftime("%H%M%S") + ".jpg", pil))

    ctx = webrtc_streamer(key="cam", video_processor_factory=Cam)

    if ctx.video_processor:
        if st.button("Capture"):
            ctx.video_processor.capture(ctx.video_processor)
            images_list.extend(ctx.video_processor.frames)
            ctx.video_processor.frames = []

# =========================================================
# History
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Filename","Width","Height","Type","Time","Device","Top5"
    ])

# =========================================================
# Run inference
# =========================================================
for fname, img in images_list:

    st.image(img, width=200)

    results = predict(img)

    labels = []
    scores = []
    texts = []

    for i,(eng, zh, score) in enumerate(results):
        text = f"{eng}({zh}) {score:.2f}%"
        texts.append(text)
        labels.append(eng)
        scores.append(score)

        if i == 0:
            st.markdown(f"<b style='color:red'>{i+1}. {text}</b>", unsafe_allow_html=True)
        else:
            st.write(f"{i+1}. {text}")

    fig, ax = plt.subplots()
    ax.barh(labels[::-1], scores[::-1])
    st.pyplot(fig)

    st.session_state.history = pd.concat([
        st.session_state.history,
        pd.DataFrame([{
            "Filename": fname,
            "Width": img.width,
            "Height": img.height,
            "Type": img.format,
            "Time": datetime.now(),
            "Device": upload_mode,
            "Top5": texts
        }])
    ], ignore_index=True)

# =========================================================
# Show history
# =========================================================
st.subheader("History")
st.dataframe(st.session_state.history)