# =============================================================================
#  TITANIC — CÓDIGO DE INFERÊNCIA EM PRODUÇÃO
#  Modelo: AttentionMLP  |  Tipo: pytorch
#  Gerado em: 2026-04-27 18:11
# =============================================================================

import json
import numpy as np
import joblib
from pathlib import Path

# Para modelos PyTorch, descomentar:
# import torch
# import torch.nn as nn

# ── Carregamento dos artefatos ─────────────────────────────────────────────────
BASE = Path(__file__).parent
with open(BASE / "config.json") as f:
    CONFIG = json.load(f)

scaler = joblib.load(BASE / "scaler.joblib")

# PyTorch:
# Reconstituir arquitetura e carregar pesos:
# from titanic_pipeline import AttentionMLP
# model = AttentionMLP(n_features=12)
# model.load_state_dict(torch.load(BASE / 'best_model.pth', map_location='cpu'))
# model.eval()

FEATURE_NAMES = ['pclass', 'sex_enc', 'age', 'sibsp', 'parch', 'log_fare', 'log_fare_per_person', 'family_size', 'is_alone', 'age_pclass', 'emb_Q', 'emb_S']


def preprocess_single(pclass, sex, age, sibsp, parch, fare, embarked):
    """
    Pré-processa um único passageiro para inferência.

    Args:
        pclass   : 1, 2 ou 3
        sex      : 'male' ou 'female'
        age      : idade em anos (float)
        sibsp    : nº de irmãos/cônjuge a bordo
        parch    : nº de pais/filhos a bordo
        fare     : tarifa paga (float)
        embarked : 'C', 'Q' ou 'S'

    Returns:
        np.ndarray de shape (1, n_features) normalizado
    """
    family_size        = sibsp + parch + 1
    is_alone           = int(family_size == 1)
    log_fare           = np.log1p(fare)
    fare_per_person    = fare / family_size
    log_fare_per_person = np.log1p(fare_per_person)
    age_pclass         = age * pclass
    sex_enc            = int(sex == "female")
    emb_Q              = int(embarked == "Q")
    emb_S              = int(embarked == "S")

    row = np.array([[
        pclass, sex_enc, age, sibsp, parch,
        log_fare, log_fare_per_person,
        family_size, is_alone, age_pclass,
        emb_Q, emb_S
    ]], dtype=np.float32)

    return scaler.transform(row)


def predict(pclass, sex, age, sibsp, parch, fare, embarked, threshold=0.5):
    """
    Realiza predição para um único passageiro.

    Returns:
        dict com 'classe', 'label', 'probabilidade_sobrevivencia'
    """
    X = preprocess_single(pclass, sex, age, sibsp, parch, fare, embarked)

    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32))
        prob = torch.softmax(out, 1)[0, 1].item()

    classe = int(prob >= threshold)
    return {
        "classe":                   classe,
        "label":                    CONFIG["class_labels"][str(classe)],
        "probabilidade_sobrevivencia": round(prob, 4),
        "threshold_usado":          threshold,
    }


# ── Exemplo de uso ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Rose: mulher, 1ª classe, 17 anos
    resultado = predict(
        pclass=1, sex="female", age=17,
        sibsp=1, parch=2, fare=263.0, embarked="S"
    )
    print("Rose:", resultado)

    # Jack: homem, 3ª classe, 20 anos
    resultado = predict(
        pclass=3, sex="male", age=20,
        sibsp=0, parch=0, fare=7.25, embarked="S"
    )
    print("Jack:", resultado)
