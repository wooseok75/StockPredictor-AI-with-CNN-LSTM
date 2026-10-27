import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")


def evaluate_classifier(X_val, y_class_val): # 분류 모델 평가 함수
    model_path = os.path.join(MODEL_DIR, "classifier.keras")
    model = tf.keras.models.load_model(model_path, compile = False)

    y_prob = model.predict(X_val, verbose=0).ravel()

    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_class_val, y_pred)
    auc = roc_auc_score(y_class_val, y_prob)
    cm = confusion_matrix(y_class_val, y_pred, labels=[0, 1])

    print(f"Val_acc: {accuracy:.4f}")
    print(f"Val_AUC: {auc:.4f}\n")

    print("CM:")
    print(cm, '\n')

    print("예측 확률 분포:")
    print(f"최솟값: {y_prob.min():.4f}")
    print(f"최댓값: {y_prob.max():.4f}")


def evaluate_regressor(X_val, y_reg_val): # 회귀 모델 평가 함수
    model_path = os.path.join(MODEL_DIR, "regressor.keras")
    model = tf.keras.models.load_model(model_path, compile = False)

    y_pred = model.predict(X_val, verbose = 0).ravel()

    mae = mean_absolute_error(y_reg_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_val, y_pred))

    # 기준 모델: 항상 0% 수익률 예측
    baseline_pred = np.zeros_like(y_reg_val)
    baseline_mae = mean_absolute_error(y_reg_val, baseline_pred)
    baseline_rmse = np.sqrt(
        mean_squared_error(y_reg_val, baseline_pred)
    )

    # 예측 수익률을 기준으로 상승/하락 방향 평가
    y_class = (y_reg_val > 0).astype(int)
    y_direction = (y_pred > 0).astype(int)

    direction_acc = accuracy_score(y_class, y_direction)
    direction_auc = roc_auc_score(y_class, y_pred)

    print(f"Validation MAE: {mae:.6f}")
    print(f"Validation RMSE: {rmse:.6f}")

    print("\n[0% 수익률 예측 Baseline]")
    print(f"Baseline MAE: {baseline_mae:.6f}")
    print(f"Baseline RMSE: {baseline_rmse:.6f}")

    print("\n[방향 예측 평가]")
    print(f"Direction Accuracy: {direction_acc:.4f}")
    print(f"Direction AUC: {direction_auc:.4f}")

    print("\n[예측 수익률 분포]")
    print(f"최솟값: {y_pred.min():.6f}")
    print(f"최댓값: {y_pred.max():.6f}")
    print(f"평균: {y_pred.mean():.6f}")


def evaluate_models():
    val_data = np.load(os.path.join(DATA_DIR, "val.npz"))

    X_val = val_data["X"]
    y_class_val = val_data["y_class"]
    y_reg_val = val_data["y_reg"]

    evaluate_classifier(X_val, y_class_val)
    evaluate_regressor(X_val, y_reg_val)

    print("[모든 모델 평가 완료!]")
