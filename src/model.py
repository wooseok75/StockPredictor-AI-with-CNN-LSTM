import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
EPOCHS = 50
BATCH_SIZE = 64


def build_classifier(input_shape):
    model = Sequential([
        Input(shape = input_shape),
        Conv1D(filters = 32, kernel_size = 3, padding = "same", activation = "relu"),
        Conv1D(filters = 64, kernel_size = 3, padding = "same", activation = "relu"),
        LSTM(64),
        Dense(64, activation = "relu"),
        Dropout(0.3),
        Dense(1, activation = "sigmoid")
    ])

    model.compile(optimizer = Adam(learning_rate=0.001), loss = "binary_crossentropy", metrics = ["accuracy", AUC()])

    return model


def build_regressor(input_shape):
    model = Sequential([
        Input(shape = input_shape),
        Conv1D(filters = 32, kernel_size = 3, padding = "same", activation = "relu"),
        Conv1D(filters = 64, kernel_size = 3, padding = "same", activation = "relu"),
        LSTM(64),
        Dense(64, activation = "relu"),
        Dropout(0.3),
        Dense(1, activation = "linear")
    ])

    model.compile(optimizer = Adam(learning_rate = 0.001), loss = "mse", metrics = ["mae"])

    return model


def train_model(model, X_train, y_train, X_val, y_val, name):
    print(f"[{name} 모델 학습 시작]\n")

    early_stopping = EarlyStopping(monitor = "val_loss", patience = 5,restore_best_weights = True)

    reduce_lr = ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 2, min_lr = 0.00001)

    history = model.fit(
        X_train,
        y_train,
        validation_data = (X_val, y_val),
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        callbacks = [early_stopping, reduce_lr],
        shuffle = False
    )

    model_path = os.path.join(MODEL_DIR, f"{name}.keras")
    model.save(model_path)

    print(f"[{name} 모델 저장 완료]")

    return history


def train_models(datasets):
    tf.keras.utils.set_random_seed(SEED)

    X_train = datasets["train"]["X"]
    X_val = datasets["val"]["X"]

    y_class_train = datasets["train"]["y_class"]
    y_class_val = datasets["val"]["y_class"]

    y_reg_train = datasets["train"]["y_reg"]
    y_reg_val = datasets["val"]["y_reg"]

    input_shape = X_train.shape[1:]

    # 분류 모델
    classifier = build_classifier(input_shape)

    classifier.summary()

    train_model(classifier, X_train, y_class_train, X_val, y_class_val, name = "classifier")

    # 회귀 모델
    regressor = build_regressor(input_shape)

    regressor.summary()

    train_model(regressor, X_train, y_reg_train, X_val, y_reg_val, name = "regressor")

    print("[모든 모델 학습 및 저장 완료!]")