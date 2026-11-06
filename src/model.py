from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 경로 설정
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# 파라미터 설정
EPOCHS = 200
BATCH_SIZE = 64
SEED = 42

# classifier 모델 생성 함수
def build_classifier(input_shape):
    classifier = Sequential([
        Input(shape = input_shape),
        Conv1D(32, 3, padding = "same", activation = "relu"),
        Conv1D(64, 3, padding = "same", activation = "relu"),
        LSTM(64),
        Dense(64, activation = "relu"),
        Dropout(0.3),
        Dense(1, activation = "sigmoid")
    ])

    classifier.compile(loss = "binary_crossentropy", optimizer = Adam(learning_rate = 0.001), metrics = ['accuracy', AUC()])

    return classifier

# regressor 모델 생성 함수
def build_regressor(input_shape):
    regressor = Sequential([
        Input(shape = input_shape),
        Conv1D(32, 3, padding = "same", activation = "relu"),
        Conv1D(64, 3, padding = "same", activation = "relu"),
        LSTM(64),
        Dense(64, activation = "relu"),
        Dropout(0.3),
        Dense(1, activation = "linear")
    ])

    regressor.compile(loss = "mse", optimizer = Adam(learning_rate = 0.001), metrics = ['mae'])

    return regressor

# 모델 학습 함수
def train_model(model, X_train, y_train, X_val, y_val, name):
    print(f"{name} 모델 학습 시작!")

    callbacks = [
        EarlyStopping(monitor = "val_loss", patience = 5, restore_best_weights = True),
        ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 2, min_lr = 1e-5)
    ]

    model.fit(
        X_train,
        y_train,
        validation_data = (X_val, y_val),
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        callbacks = callbacks,
        shuffle = True
    )

    MODEL_DIR.mkdir(exist_ok = True)
    model.save(MODEL_DIR / f"{name}.keras")

    print(f"{name} 모델 저장 완료!\n")

# 실행 함수
def train_models(train_set, val_set):
    X_train, y_class_train, y_reg_train = train_set['X'], train_set['y_class'], train_set['y_reg']
    X_val, y_class_val, y_reg_val = val_set['X'], val_set['y_class'], val_set['y_reg']

    tf.keras.utils.set_random_seed(SEED)
    
    input_shape = X_train.shape[1:]

    classifier = build_classifier(input_shape)
    regressor = build_regressor(input_shape)

    train_model(classifier, X_train, y_class_train, X_val, y_class_val, "classifier")
    train_model(regressor, X_train, y_reg_train, X_val, y_reg_val, "regressor")

    print("모든 모델 학습 완료!")