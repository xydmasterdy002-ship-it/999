# ========================= FINAL STABLE VERSION (Modified UI) =========================
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
    def __init__(self, num_classes=100, include_top=True, pooling=None, pretrained=False):
        super().__init__()
        self.include_top = include_top
        self.pooling = pooling

        # 與訓練完全一致
        self.base_model = models.resnet101(
            weights=models.ResNet101_Weights.DEFAULT if pretrained else None
        )

        if include_top:
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        else:
            self.base_model.fc = nn.Identity()

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)

        if self.include_top:
            x = self.base_model.fc(x)
        else:
            if self.pooling == 'avg':
                x = torch.mean(x, dim=-1, keepdim=True)
            elif self.pooling == 'max':
                x, _ = torch.max(x, dim=-1, keepdim=True)
        return x

class GoogLeNetRobust(nn.Module):
    def __init__(self, num_classes=100, device=None):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.googlenet(weights=None, aux_logits=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.aux1.fc2 = nn.Linear(self.model.aux1.fc2.in_features, num_classes)
        self.model.aux2.fc2 = nn.Linear(self.model.aux2.fc2.in_features, num_classes)
        self.model.to(self.device).eval()

    def load_weights(self, path, strict=False):
        state_dict = torch.load(path, map_location=self.device)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=strict)
        self.model.to(self.device).eval()
        return self

    def forward(self, x):
        out = self.model(x.to(self.device))
        if isinstance(out, tuple):
            out = out[0]
        return out

# =========================================================
# Load Model
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
        return model.model
    else:
        # 與訓練設定對齊
        model = ResNet101Custom(
            num_classes=num_classes,
            include_top=True,
            pooling=None,
            pretrained=False  # 推論時不需要再載 pretrained
        )

    state_dict = torch.load(info["path"], map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # ResNet 使用嚴格對齊（因為結構已完全一致）
    if model_name == "ResNet101":
        model.load_state_dict(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

# =========================================================
# UI (Reordered v2)
# =========================================================
st.title("ImageNet100 Prediction System")

# 1. Select model first
model_name = st.selectbox("Select Model", ["MobileNetV1","GoogLeNet","ResNet101"])
model = load_model(model_name)

# 2. Upload image
files = st.file_uploader("上傳圖片", type=["jpg","png","jpeg"], accept_multiple_files=True)

# 3. History init
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Filename","Time","Model",
        "1st class","1st P(%)",
        "2nd class","2nd P(%)",
        "3rd class","3rd P(%)",
        "4th class","4th P(%)",
        "5th class","5th P(%)"
    ])

# =========================================================
# Prediction
# =========================================================
@st.cache_data
def translate_cached(text):
    return translator.translate(text)


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
# Inference + History display (同步更新)
# =========================================================
suffix = ["st","nd","rd","th","th"]

if files:
    for f in files:
        img = Image.open(f).convert("RGB")

        # ===== 推論 =====
        results = predict(img)

        # ===== 顯示圖片 =====
        st.image(img, width=200)

        labels = [r[0] for r in results]
        scores = [r[2] for r in results]

        # ===== 顯示排名 =====
        for i, (eng, zh, score) in enumerate(results):
            text = f"{eng}({zh}) {score:.2f}%"
            if i==0:
                st.markdown(f"<b style='color:red'>{i+1}. {text}</b>", unsafe_allow_html=True)
            else:
                st.write(f"{i+1}. {text}")

        # ===== 長條圖 =====
        fig, ax = plt.subplots()
        ax.barh(labels[::-1], scores[::-1])
        st.pyplot(fig)

        # ===== 更新 history =====
        row = {"Filename": f.name, "Time": datetime.now(), "Model": model_name}
        for i,(eng, zh, score) in enumerate(results):
            row[f"{i+1}{suffix[i]} class"] = f"{eng}({zh})"
            row[f"{i+1}{suffix[i]} P(%)"] = round(score,2)

        st.session_state.history = pd.concat([
            st.session_state.history,
            pd.DataFrame([row])
        ], ignore_index=True)

# ===== History (永遠顯示在上方，且即時更新) =====
st.subheader("History")
st.dataframe(st.session_state.history)

# =========================================================
# =========================================================
@st.cache_data
def translate_cached(text):
    return translator.translate(text)


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
# Inference (after history)
# =========================================================
suffix = ["st","nd","rd","th","th"]

if files:
    for f in files:
        img = Image.open(f).convert("RGB")
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

        row = {"Filename": f.name, "Time": datetime.now(), "Model": model_name}
        for i,(eng, zh, score) in enumerate(results):
            row[f"{i+1}{suffix[i]} class"] = f"{eng}({zh})"
            row[f"{i+1}{suffix[i]} P(%)"] = round(score,2)

        st.session_state.history = pd.concat([
            st.session_state.history,
            pd.DataFrame([row])
        ], ignore_index=True)
