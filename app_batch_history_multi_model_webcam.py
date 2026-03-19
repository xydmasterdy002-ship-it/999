# ========================= FINAL STABLE VERSION (修正版) =========================
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
# Model config
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
# Download model
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
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
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
            conv_bn(3,32,2),
            conv_dw(32,64,1),
            conv_dw(64,128,2),
            conv_dw(128,128,1),
            conv_dw(128,256,2),
            conv_dw(256,256,1),
            conv_dw(256,512,2),
            *[conv_dw(512,512,1) for _ in range(5)],
            conv_dw(512,1024,2),
            conv_dw(1024,1024,1),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(1024,num_classes)

    def forward(self,x):
        x=self.model(x)
        x=x.view(x.size(0),-1)
        return self.fc(x)

class ResNet101Custom(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        base = models.resnet101(weights=None)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.base = base

    def forward(self, x):
        return self.base(x)

# ====================== Robust GoogLeNet Wrapper ======================
class GoogLeNetRobust(nn.Module):
    def __init__(self, num_classes=100, pretrained=False, use_aux=True, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 建立基礎模型
        self.model = models.googlenet(
            weights=None if not pretrained else models.GoogLeNet_Weights.DEFAULT,
            aux_logits=use_aux
        )
        # 修改 fc 層
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        if use_aux:
            self.model.aux1.fc2 = nn.Linear(self.model.aux1.fc2.in_features, num_classes)
            self.model.aux2.fc2 = nn.Linear(self.model.aux2.fc2.in_features, num_classes)

        self.model.to(self.device)
        self.model.eval()

    def load_weights(self, path, strict=False, use_dataparallel=True):
        state_dict = torch.load(path, map_location=self.device)
        # 自動去掉 module. 前綴
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        if use_dataparallel and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.model.load_state_dict(state_dict, strict=strict)
        self.model.to(self.device).eval()
        return self

    def forward(self, x):
        out = self.model(x.to(self.device))
        if isinstance(out, tuple):
            out = out[0]
        return out

# =========================================================
# Load Model (統一接口)
# =========================================================
@st.cache_resource
def load_model(model_name):
    info = MODEL_FILES[model_name]
    download_model(info["file_id"], info["path"])

    if model_name=="MobileNetV1":
        model = MobileNetV1(num_classes)
    elif model_name=="GoogLeNet":
        model = GoogLeNetRobust(num_classes=num_classes)
        model.load_weights(info["path"], strict=False)
        return model.model  # 回傳內部 model 保持接口一致
    else:
        model = ResNet101Custom(num_classes)

    # MobileNet / ResNet 載入權重
    state_dict = torch.load(info["path"], map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

# =========================================================
# UI
# =========================================================
st.title("ImageNet100 Prediction System")
model_name = st.selectbox("Select Model", ["MobileNetV1","GoogLeNet","ResNet101"])
model = load_model(model_name)

# =========================================================
# Translate cache
# =========================================================
@st.cache_data
def translate_cached(text):
    return translator.translate(text)

# =========================================================
# Prediction
# =========================================================
def predict(image):
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = F.softmax(model(x), dim=1)
        p, idx = prob.topk(5)

    results=[]
    for i in range(5):
        label = idx_to_label[idx[0][i].item()]
        eng = label.split(",")[0]
        zh = translate_cached(eng)
        score = p[0][i].item()*100
        results.append((eng, zh, score))
    return results

# =========================================================
# Upload Mode
# =========================================================
mode = st.radio("選擇輸入方式", [
    "選電腦/手機內照片",
    "啟動照相機拍攝"
])

images=[]

if mode=="選電腦/手機內照片":
    files=st.file_uploader("上傳圖片", type=["jpg","png","jpeg"], accept_multiple_files=True)
    if files:
        for f in files:
            images.append((f.name, Image.open(f).convert("RGB")))

elif mode=="啟動照相機拍攝":
    class Cam(VideoProcessorBase):
        def __init__(self):
            self.frame=None

        def recv(self, frame):
            self.frame=frame
            return frame

        def capture(self):
            if self.frame is None:
                return None
            img=self.frame.to_ndarray(format="bgr24")
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ctx = webrtc_streamer(key="cam", video_processor_factory=Cam)
    if ctx.video_processor and st.button("拍照"):
        img = ctx.video_processor.capture()
        if img:
            images.append((f"cam_{datetime.now().strftime('%H%M%S')}.jpg", img))
        else:
            st.warning("尚未取得影像")

# =========================================================
# History
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Filename","Time","Model",
        "1st class","1st P(%)",
        "2nd class","2nd P(%)",
        "3rd class","3rd P(%)",
        "4th class","4th P(%)",
        "5th class","5th P(%)"
    ])

suffix = ["st","nd","rd","th","th"]

# =========================================================
# Inference
# =========================================================
for fname,img in images:
    st.image(img, width=200)
    results = predict(img)

    labels = [r[0] for r in results]
    scores = [r[2] for r in results]

    for i, (eng, zh, score) in enumerate(results):
        text = f"{eng}({zh}) {score:.2f}%"
        if i==0:
            st.markdown(f"<b style='color:red'>{i+1}. {text}</b>", unsafe_allow_html=True)
        else:
            st.write(f"{i+1}. {text}")

    fig, ax = plt.subplots()
    ax.barh(labels[::-1], scores[::-1])
    st.pyplot(fig)

    row = {"Filename": fname, "Time": datetime.now(), "Model": model_name}
    for i,(eng, zh, score) in enumerate(results):
        row[f"{i+1}{suffix[i]} class"] = f"{eng}({zh})"
        row[f"{i+1}{suffix[i]} P(%)"] = round(score,2)

    st.session_state.history = pd.concat([
        st.session_state.history,
        pd.DataFrame([row])
    ], ignore_index=True)

# =========================================================
# Show History
# =========================================================
st.subheader("History")
st.dataframe(st.session_state.history)