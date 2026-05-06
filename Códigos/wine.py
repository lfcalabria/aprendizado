import os
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="🍷 Wine – Multi Model", layout="centered")
st.title("🍷 Wine Quality – Classificação (4 abordagens)")
st.caption("Escolha o ambiente, informe as variáveis e classifique.")

# ==========================================================
# PASTAS DOS AMBIENTES
# ==========================================================
ENV_DIRS = {
    "Clássico (Sklearn/Torch)": "wine_classico",
    "AutoML (PyCaret)": "wine_pycaret",
    "AutoML (FLAML)": "wine_flaml",
    "AutoML (AutoGluon)": "wine_autogluon",
}

def load_meta(env_folder: str) -> dict:
    meta_path = os.path.join(env_folder, "meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_artifact(env_folder: str, meta: dict):
    kind = meta["model_kind"]
    artifact_path = meta["artifact_path"]

    if kind == "sklearn":
        return joblib.load(os.path.join(env_folder, artifact_path))

    if kind == "pycaret":
        from pycaret.classification import load_model as pycaret_load_model
        # artifact_path é prefixo, sem .pkl
        return pycaret_load_model(os.path.join(env_folder, artifact_path))

    if kind == "autogluon":
        from autogluon.tabular import TabularPredictor
        return TabularPredictor.load(os.path.join(env_folder, artifact_path))

    if kind == "torch":
        # retorna dict com caminhos (bundle)
        bundle = artifact_path
        preproc = joblib.load(os.path.join(env_folder, bundle["preprocessor"]))
        le = joblib.load(os.path.join(env_folder, bundle["label_encoder"]))
        with open(os.path.join(env_folder, bundle["torch_info"]), "r", encoding="utf-8") as f:
            info = json.load(f)
        state_path = os.path.join(env_folder, bundle["state_dict"])
        return {"preprocessor": preproc, "label_encoder": le, "info": info, "state_path": state_path}

    raise ValueError(f"model_kind não suportado: {kind}")

def predict(env_folder: str, meta: dict, artifact, X_input: pd.DataFrame):
    kind = meta["model_kind"]

    if kind == "sklearn":
        return artifact.predict(X_input)[0]

    if kind == "pycaret":
        from pycaret.classification import predict_model
        out = predict_model(artifact, data=X_input)
        return out["prediction_label"].iloc[0]

    if kind == "autogluon":
        return artifact.predict(X_input).iloc[0]

    if kind == "torch":
        import torch
        import torch.nn as nn

        preproc = artifact["preprocessor"]
        le = artifact["label_encoder"]
        info = artifact["info"]

        # reconstruir arquitetura
        class TorchMLP(nn.Module):
            def __init__(self, in_dim, n_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, n_classes)
                )
            def forward(self, x):
                return self.net(x)

        class TorchCNN(nn.Module):
            def __init__(self, in_dim, n_classes):
                super().__init__()
                self.conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self.fc = nn.Linear(32 * in_dim, n_classes)
            def forward(self, x):
                x = x.unsqueeze(1)
                x = torch.relu(self.conv(x))
                x = x.view(x.size(0), -1)
                return self.fc(x)

        class TorchRNN(nn.Module):
            def __init__(self, in_dim, n_classes):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
                self.fc = nn.Linear(64, n_classes)
            def forward(self, x):
                x = x.unsqueeze(-1)
                _, (h, _) = self.lstm(x)
                return self.fc(h[-1])

        arch = info["arch"]
        in_dim = info["input_dim"]
        n_classes = info["n_classes"]

        if arch == "Torch_MLP":
            mdl = TorchMLP(in_dim, n_classes)
        elif arch == "Torch_CNN":
            mdl = TorchCNN(in_dim, n_classes)
        else:
            mdl = TorchRNN(in_dim, n_classes)

        mdl.load_state_dict(torch.load(artifact["state_path"], map_location="cpu"))
        mdl.eval()

        Xp = preproc.transform(X_input)
        Xt = torch.tensor(Xp, dtype=torch.float32)
        y_hat = mdl(Xt).argmax(1).item()

        # retorna label original (quality real)
        return int(le.inverse_transform([y_hat])[0])

    raise ValueError(f"model_kind não suportado: {kind}")

# ==========================================================
# UI
# ==========================================================
env_name = st.selectbox("Escolha o ambiente:", list(ENV_DIRS.keys()))
env_folder = ENV_DIRS[env_name]

meta = load_meta(env_folder)
if not meta:
    st.error(f"meta.json não encontrado em {env_folder}.")
    st.stop()

features = meta.get("features", [])
if not features:
    st.error("meta.json não contém 'features'.")
    st.stop()

st.subheader("Variáveis")
st.write(features)

user_data = {}
for feat in features:
    user_data[feat] = st.number_input(feat, value=0.0, format="%.6f")

X_input = pd.DataFrame([user_data])

artifact = load_artifact(env_folder, meta)

if st.button("Classificar"):
    y_pred = predict(env_folder, meta, artifact, X_input)
    st.success(f"✅ Classe prevista: **{y_pred}**")

    if "original_classes" in meta:
        st.caption(f"Classes originais: {meta['original_classes']}")