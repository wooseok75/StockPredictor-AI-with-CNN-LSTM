import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# 모델 성능 평가를 위해 시드 고정
np.random.seed(42)
tf.random.set_seed(42)

# 데이터 가져오기
train_data = np.load(os.path.join(DATA_DIR, 'train.npz'))
val_data = np.load(os.path.join(DATA_DIR, 'val.npz'))
test_data = np.load(os.path.join(DATA_DIR, 'test.npz'))

X_train, y_train = train_data['X'], train_data['y']
X_val, y_val = val_data['X'], val_data['y']
X_test, y_test = test_data['X'], test_data['y']

input_shape = X_train.shape[1:]

# 모델 설정
model = Sequential([
    Input(shape = input_shape),

    Conv1D(
        filters = 32,
        kernel_size = 3,
        padding = 'same',
        activation = 'relu'
    ),

    Conv1D(
        filters = 64,
        kernel_size = 3,
        padding = 'same',
        activation = 'relu'
    ),

    LSTM(64),

    Dense(64, activation = 'relu'),

    Dropout(0.3),

    Dense(1, activation = 'sigmoid')
])

model.summary()

model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy', tf.keras.metrics.AUC(name="auc")])

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 2, min_lr = 0.00001)

history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 50, batch_size = 64, callbacks = [early_stopping, reduce_lr], shuffle = False)

# 성능 확인
print()
print("==== Validation Evaluation =====")

val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose = 0)

print(f"Val_loss: {val_loss:.4f}")
print(f"Val_acc: {val_acc:.4f}")
print(f"Val_AUC: {val_auc:.4f}")

print()
print("==== Test Evaluation ====")

test_loss, test_accuracy, test_auc = model.evaluate(X_test, y_test, verbose=0)

print(f"Test_loss: {test_loss:.4f}")
print(f"Test_acc: {test_accuracy:.4f}")
print(f"Test_AUC: {test_auc:.4f}")

y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print()
print("==== Confusion Matrix ====")

cm = confusion_matrix(y_test, y_pred)

print(cm)

# 모델 저장
model_path = os.path.join(MODEL_DIR, "model.keras")

model.save(model_path)

print()
print(f"[모델 저장 완료] {model_path}")