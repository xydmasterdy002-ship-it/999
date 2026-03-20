# ========================= FINAL STABLE VERSION (微調，VGG/EfficientNet/ViT top-1 修正) =========================
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from deep_translator import GoogleTranslator
import pandas as pd
import gdown

# =========================================================
# 模型檔案設定
# =========================================================
MODEL_FILES = {
    "MobileNet-V1": {"file_id": "1FqpeWNvrSHZk0o0Cka2o_BIXyjPgosMi", "path": "MobileNet-V1_imagenet100.pth"},
    "GoogLeNet": {"file_id": "1SrBz2SQ1VcyEItV-pX6HozJzKkHtJ2AR", "path": "googlenet_imagenet100.pth"},
    "ResNet-101": {"file_id": "1RnSGWxh99VG3kb_LN-cx7rp1VWmEFEWs", "path": "Resnet-101_imagenet100.pth"},
    "EfficientNet-B0": {"file_id": "10P8pjuyQqrXiCZJhujOKesS3263-65iC", "path": "EfficientNet-B0_imagenet100.pth"},
    "VGG-16": {"file_id": "1LE5ghQK-uDipL1uh35LazL1j2ivVgRnh", "path": "VGG-16_imagenet100.pth"},
    "ViT": {"file_id": "1KEZOMHPp0KTXLSMP1jiYVOh0-cv8hMxI", "path": "ViT_imagenet100.pth"}
}

LABEL_PATH = "Labels.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translator = GoogleTranslator(source="en", target="zh-TW")

# =========================================================
# 下載模型函數
# =========================================================
def download_model(file_id, output):
    if not os.path.exists(output):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"Downloading {output}..."):
            gdown.download(url, output, quiet=False)

# =========================================================
# 讀取標籤
# =========================================================
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    wnid_to_label = json.load(f)

class_names = sorted(wnid_to_label.keys())
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
idx_to_label = {idx: wnid_to_label[cls] for cls, idx in class_to_idx.items()}
num_classes = len(idx_to_label)

# =========================================================
# Image transform
# =========================================================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# =========================================================
# 建立自訂模型類別
# =========================================================
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                                 nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))
        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                                 nn.BatchNorm2d(inp), nn.ReLU6(inplace=True),
                                 nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                                 nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))
        self.model = nn.Sequential(conv_bn(3,32,2),
                                   conv_dw(32,64,1),
                                   conv_dw(64,128,2),
                                   conv_dw(128,128,1),
                                   conv_dw(128,256,2),
                                   conv_dw(256,256,1),
                                   conv_dw(256,512,2),
                                   *[conv_dw(512,512,1) for _ in range(5)],
                                   conv_dw(512,1024,2),
                                   conv_dw(1024,1024,1),
                                   nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(1024,num_classes)
    def forward(self,x):
        x=self.model(x)
        x=x.view(x.size(0),-1)
        return self.fc(x)

class ResNet101Custom(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = models.resnet101(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self,x):
        return self.model(x)

class GoogLeNetRobust(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = models.googlenet(weights=None, aux_logits=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.aux1.fc2 = nn.Linear(self.model.aux1.fc2.in_features, num_classes)
        self.model.aux2.fc2 = nn.Linear(self.model.aux2.fc2.in_features, num_classes)
    def load_weights(self,path):
        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace("module.",""):v for k,v in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=False)
        return self
    def forward(self,x):
        out = self.model(x)
        if isinstance(out,tuple):
            out = out[0]
        return out

# =========================================================
# EfficientNet / VGG / ViT 自訂 class
# =========================================================
class EfficientNetB0Custom(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
    def forward(self,x):
        return self.model(x)

class VGG16Custom(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vgg16(weights=None)
        self.model.classifier[6] = nn.Linear(4096,num_classes)
    def forward(self,x):
        return self.model(x)

class ViTB16Custom(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vit_b_16(weights=None)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
    def forward(self,x):
        return self.model(x)

# =========================================================
# 載入模型函數
# =========================================================
@st.cache_resource
def load_model(model_name):
    info = MODEL_FILES[model_name]
    download_model(info["file_id"], info["path"])

    if model_name=="MobileNet-V1":
        model = MobileNetV1(num_classes)
    elif model_name=="GoogLeNet":
        model = GoogLeNetRobust(num_classes)
        model.load_weights(info["path"])
        return model.model
    elif model_name=="ResNet-101":
        model = ResNet101Custom(num_classes)
    elif model_name=="EfficientNet-B0":
        model = EfficientNetB0Custom(num_classes)
    elif model_name=="VGG-16":
        model = VGG16Custom(num_classes)
    elif model_name=="ViT":
        model = ViTB16Custom(num_classes)

    # 載入權重
    state_dict = torch.load(info["path"], map_location=device)
    new_state_dict = {k.replace("module.",""):v for k,v in state_dict.items()}
    if model_name in ["EfficientNet-B0","VGG-16","ViT"]:
        model.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=True)

    model.to(device).eval()
    return model

# =========================================================
# UI
# =========================================================
st.title("ImageNet100 Prediction System")

model_name = st.selectbox("Select Model", list(MODEL_FILES.keys()))
model = load_model(model_name)

files = st.file_uploader("上傳圖片", type=["jpg","png","jpeg"], accept_multiple_files=True)

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Filename","Height(px)","Width(px)","Model",
        "1st class","1st P(%)",
        "2nd class","2nd P(%)",
        "3rd class","3rd P(%)",
        "4th class","4th P(%)",
        "5th class","5th P(%)"
    ])

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

suffix = ["st","nd","rd","th","th"]

if files:
    temp_results = []
    for f in files:
        img = Image.open(f).convert("RGB")
        results = predict(img)
        temp_results.append((f.name, img, results))
        width, height = img.size
        row = {"Filename": f.name, "Height(px)": height, "Width(px)": width, "Model": model_name}
        for i,(eng, zh, score) in enumerate(results):
            row[f"{i+1}{suffix[i]} class"] = f"{eng}({zh})"
            row[f"{i+1}{suffix[i]} P(%)"] = round(score,2)
        st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([row])], ignore_index=True)

st.subheader("History")
st.dataframe(st.session_state.history)

if files:
    for fname, img, results in temp_results:
        st.image(img, width=200)
        for i,(eng, zh, score) in enumerate(results):
            text = f"{eng}({zh}) {score:.2f}%"
            if i==0:
                st.markdown(f"<b style='color:red'>{i+1}. {text}</b>", unsafe_allow_html=True)
            else:
                st.write(f"{i+1}. {text}")
        fig, ax = plt.subplots()
        ax.barh([r[0] for r in results][::-1], [r[2] for r in results][::-1])
        st.pyplot(fig)