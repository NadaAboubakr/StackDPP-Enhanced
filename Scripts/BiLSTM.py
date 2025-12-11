import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, Flatten, GlobalAveragePooling1D
from keras.layers import Attention
from keras import regularizers
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score

# Load dataset
import os
file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Features', 'rf452.npz')
data = np.load(file_path, allow_pickle=True)
X = data['X']
y = data['y']
test_X = data['test_X']
test_y = data['test_y']

# Ensure correct shape (samples, time_steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))

# Model v2 — Regularized BiLSTM + Dropout + Attention
inputs = Input(shape=(X.shape[1], 1))

x = Bidirectional(LSTM(196, return_sequences=True,
                       kernel_regularizer=regularizers.l2(1e-4)))(inputs)
x = Dropout(0.4)(x)

x = Bidirectional(LSTM(128, return_sequences=True,
                       kernel_regularizer=regularizers.l2(1e-4)))(x)
x = Dropout(0.4)(x)

att = Attention()([x, x])

# Flatten the sequence dimension using GlobalAveragePooling1D
x = GlobalAveragePooling1D()(att)

x = Dense(128, activation='relu',
          kernel_regularizer=regularizers.l2(1e-4))(x)
x = Dropout(0.3)(x)

outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train
history = model.fit(
    X, y,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=1
)

# Evaluate
pred = model.predict(test_X)
pred_binary = (pred > 0.5).astype(int)

ACC = accuracy_score(test_y, pred_binary)
SN = recall_score(test_y, pred_binary)
SP = recall_score(test_y, pred_binary, pos_label=0)
MCC = matthews_corrcoef(test_y, pred_binary)

print("\n===== TEST RESULTS (BiLSTM v2) =====")
print("ACC:", ACC)
print("SN :", SN)
print("SP :", SP)
print("MCC:", MCC)