import librosa
from librosa import display
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import librosa.feature

path = 'E:/2024暑假\大作业\pythonProject6/archive\Emotions'
lst = []
i = -2
start_time = time.time()

for subdir, dirs, files in os.walk(path):
    i = i + 1
    print(subdir)
    print(i)
    for file in files:
        # Load librosa array, obtain mfcss, add them to array and then to list.
        X, sample_rate = librosa.load(os.path.join(subdir, file), res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_fft=4096, hop_length=256, n_mfcc=40).T, axis=0)
        arr = mfccs, i
        lst.append(arr)  # Here we append the MFCCs to our list.

print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))

file1 = 'E:/2024暑假\大作业\pythonProject6/archive\Emotions/Neutral/03-02-01-01-02-02-20.wav'
signal1, sample_rate = librosa.load(file1, sr=22050)

file2 = 'E:/2024暑假\大作业\pythonProject6/archive\Emotions/Neutral/1007_WSI_NEU_XX.wav'
signal2, sample_rate = librosa.load(file2, sr=22050)

file3 = 'E:/2024暑假\大作业\pythonProject6/archive\Emotions/Neutral/n01.wav'
signal3, sample_rate = librosa.load(file3, sr=22050)

file4 = 'E:/2024暑假\大作业\pythonProject6/archive\Emotions/Neutral/YAF_vote_neutral.wav'
signal4, sample_rate = librosa.load(file4, sr=22050)

emotion = 'Neutral'

fig = plt.figure(figsize=(15, 8))
# WAVEFORM
# display waveform
plt.subplot(2, 2, 1)
librosa.display.waveshow(y=signal1, sr=sample_rate, alpha=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("RAVDESS Waveform " + emotion)

plt.subplot(2, 2, 2)
librosa.display.waveshow(y=signal2, sr=sample_rate, alpha=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("CREMA-D Waveform " + emotion)

fig = plt.figure(figsize=(15, 8))
plt.subplot(2, 2, 3)
librosa.display.waveshow(y=signal3, sr=sample_rate, alpha=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("SAVEE Waveform " + emotion)

plt.subplot(2, 2, 4)
librosa.display.waveshow(y=signal4, sr=sample_rate, alpha=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("TESS Waveform " + emotion)

fft1 = np.fft.fft(signal1)
fft2 = np.fft.fft(signal2)
fft3 = np.fft.fft(signal3)
fft4 = np.fft.fft(signal4)

# calculate abs values on complex numbers to get magnitude
spectrum1 = np.abs(fft1)
spectrum2 = np.abs(fft2)
spectrum3 = np.abs(fft3)
spectrum4 = np.abs(fft4)

# create frequency variable
f1 = np.linspace(0, sample_rate, len(spectrum1))
f2 = np.linspace(0, sample_rate, len(spectrum2))
f3 = np.linspace(0, sample_rate, len(spectrum3))
f4 = np.linspace(0, sample_rate, len(spectrum4))

# take half of the spectrum and frequency
left_spectrum1 = spectrum1[:int(len(spectrum1) / 2)]
left_f1 = f1[:int(len(spectrum1) / 2)]
# take half of the spectrum and frequency
left_spectrum2 = spectrum2[:int(len(spectrum2) / 2)]
left_f2 = f2[:int(len(spectrum2) / 2)]
# take half of the spectrum and frequency
left_spectrum3 = spectrum3[:int(len(spectrum3) / 2)]
left_f3 = f3[:int(len(spectrum3) / 2)]
# take half of the spectrum and frequency
left_spectrum4 = spectrum4[:int(len(spectrum4) / 2)]
left_f4 = f4[:int(len(spectrum4) / 2)]

fig = plt.figure(figsize=(8, 10))
plt.subplot(2, 2, 1)
# plot spectrum
plt.plot(left_f1, left_spectrum1, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("RAVDESS  Power spectrum " + emotion)

plt.subplot(2, 2, 2)
# plot spectrum
plt.plot(left_f2, left_spectrum2, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("CREMA-D Power spectrum " + emotion)

fig = plt.figure(figsize=(8, 10))

plt.subplot(2, 2, 3)
plt.plot(left_f3, left_spectrum3, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("SAVEE Power spectrum " + emotion)

plt.subplot(2, 2, 4)
plt.plot(left_f4, left_spectrum4, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("TESS Power spectrum " + emotion)

hop_length = 256  # in num. of samples
n_fft = 4096  # window in num. of samples

# calculate duration hop length and window in seconds
hop_length_duration = float(hop_length) / sample_rate
n_fft_duration = float(n_fft) / sample_rate

print("STFT hop length duration is: {}s".format(hop_length_duration))
print("STFT window duration is: {}s".format(n_fft_duration))

# perform stft
stft1 = librosa.stft(signal1, n_fft=n_fft, hop_length=hop_length)
stft2 = librosa.stft(signal2, n_fft=n_fft, hop_length=hop_length)
stft3 = librosa.stft(signal3, n_fft=n_fft, hop_length=hop_length)
stft4 = librosa.stft(signal4, n_fft=n_fft, hop_length=hop_length)

# calculate abs values on complex numbers to get magnitude
spectrogram1 = np.abs(stft1)
spectrogram2 = np.abs(stft2)
spectrogram3 = np.abs(stft3)
spectrogram4 = np.abs(stft4)

# display spectrogram


fig = plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
librosa.display.specshow(spectrogram1, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("RAVDESS Spectrogram " + emotion)

plt.subplot(2, 2, 2)
librosa.display.specshow(spectrogram2, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("CREMA-D Spectrogram " + emotion)

fig = plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 3)
librosa.display.specshow(spectrogram3, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("SAVEE Spectrogram " + emotion)

plt.subplot(2, 2, 4)
librosa.display.specshow(spectrogram4, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.title("TESS Spectrogram " + emotion)

log_spectrogram1 = librosa.amplitude_to_db(spectrogram1)
log_spectrogram2 = librosa.amplitude_to_db(spectrogram2)
log_spectrogram3 = librosa.amplitude_to_db(spectrogram3)
log_spectrogram4 = librosa.amplitude_to_db(spectrogram4)

fig = plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
librosa.display.specshow(log_spectrogram1, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("RAVDESS Spectogramm (dB) " + emotion)

plt.subplot(2, 2, 2)
librosa.display.specshow(log_spectrogram2, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("CREMA-D Spectogramm (dB) " + emotion)

fig = plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 3)
librosa.display.specshow(log_spectrogram3, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("SAVEE Spectogramm (dB) " + emotion)

plt.subplot(2, 2, 4)
librosa.display.specshow(log_spectrogram4, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar(format="%+2.0f dB")
plt.title("TESS Spectogramm (dB) " + emotion)

# MFCCs
# extract 13 MFCCs
MFCCs1 = librosa.feature.mfcc(y=signal1, sr=sample_rate, n_fft=4096, hop_length=256, n_mfcc=40)
MFCCs2 = librosa.feature.mfcc(y=signal2, sr=sample_rate, n_fft=4096, hop_length=256, n_mfcc=40)
MFCCs3 = librosa.feature.mfcc(y=signal3, sr=sample_rate, n_fft=4096, hop_length=256, n_mfcc=40)
MFCCs4 = librosa.feature.mfcc(y=signal4, sr=sample_rate, n_fft=4096, hop_length=256, n_mfcc=40)

# display MFCCs
hop_length = 256

fig = plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
librosa.display.specshow(MFCCs1, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("RAVDESS MFCCs " + emotion)

plt.subplot(2, 2, 2)
librosa.display.specshow(MFCCs2, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("CREMA-D MFCCs " + emotion)

fig = plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 3)
librosa.display.specshow(MFCCs3, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("SAVEE MFCCs " + emotion)

plt.subplot(2, 2, 4)
librosa.display.specshow(MFCCs4, sr=sample_rate, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC coefficients")
plt.colorbar()
plt.title("TESS MFCCs " + emotion)

X, y = zip(*lst)
import numpy as np

X = np.asarray(X)
y = np.asarray(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# As always we need to expand the dimensions, so we can input the data to NN.
x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.optimizers import Adam

# 假设 x_traincnn, x_testcnn, y_train, y_test 已经被正确定义和预处理
# 这里只是示例，您需要根据实际数据调整 input_shape

# Simple model
model = Sequential([
    # input layer 假设为 (40, 1) 的形状，这里不需要额外指定 Flatten 因为已经是二维
    Flatten(input_shape=(40, )),

    # 1st dense layer
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
    Dropout(0.5),

    # 2nd dense layer
    Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.5),

    # 3rd dense layer
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.3),

    # output layer
    Dense(10, activation='softmax')
])

# compile model
optimiser = Adam(learning_rate=0.00001)
model.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 假设 x_traincnn, x_testcnn, y_train, y_test 已经被正确定义和预处理
# 训练模型
cnnhistory = model.fit(x_traincnn, y_train, batch_size=64, epochs=700, validation_data=(x_testcnn, y_test))

# 绘制训练和验证的损失
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 绘制训练和验证的准确率
plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.title('Accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

from sklearn.metrics import classification_report
import numpy as np

# 假设 model 是你的模型对象，x_testcnn 是你的测试数据，y_test 是你的真实标签
predictions = model.predict(x_testcnn)
predicted_classes = np.argmax(predictions, axis=1)
y_test = y_test.astype(int)  # 确保 y_test 是整数类型

report = classification_report(y_test, predicted_classes)
print(report)


model_name = 'EmotionClassificationModel.h5'
save_dir = 'E:/2024暑假/大作业/pythonProject6/models'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
