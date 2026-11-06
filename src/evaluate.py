from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error, roc_auc_score

# 경로 설정
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"

# classifier 평가 함수
def evaluate_classifier(X_val, y_true):
    model = tf.keras.models.load_model(MODEL_DIR / "classifier.keras", compile = False)

    y_prob = model.predict(X_val, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n[분류 모델 - Validation]")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print("혼동행렬 (0 = 하락/유지, 1 = 상승):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

# regressor 평가 함수
def evaluate_regressor(X_val, y_true):
    model = tf.keras.models.load_model(MODEL_DIR / "regressor.keras", compile=False)

    y_pred = model.predict(X_val, verbose=0).ravel()
    baseline = np.zeros_like(y_true)

    print("\n[회귀 모델 - Validation]")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.6f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.6f}")
    print(f"baseline MAE: {mean_absolute_error(y_true, baseline):.6f}")
    print(f"baseline RMSE: {np.sqrt(mean_squared_error(y_true, baseline)):.6f}")
    print(f"방향 Accuracy: {accuracy_score(y_true > 0, y_pred > 0):.4f}")

# 실행 함수
def evaluate_models():
    val_set = np.load(DATA_DIR / "val.npz")

    X_val, y_class_val, y_reg_val = val_set['X'], val_set['y_class'], val_set['y_reg']

    evaluate_classifier(X_val, y_class_val)
    evaluate_regressor(X_val, y_reg_val)

    print("모든 모델 평가 완료!")

# 파일 직접 실행
if __name__ == "__main__":
    evaluate_models()
    