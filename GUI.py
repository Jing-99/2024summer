import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# 加载你的模型
model = load_model('E:/2024暑假/大作业/pythonProject6/models/EmotionClassificationModel.h5')


# 定义处理音频的函数
def classify_emotion(audio_path):
    # 加载音频文件
    signal, sample_rate = librosa.load(audio_path, sr=22050)

    # 提取MFCC特征
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=4096, hop_length=256, n_mfcc=40).T, axis=0)

    # 扩展维度以适应模型输入
    mfccs = np.expand_dims(mfccs, axis=0)

    # 进行预测
    prediction = model.predict(mfccs)
    predicted_class = np.argmax(prediction, axis=1)

    # 映射情感标签（这里需要你根据实际情况修改）
    emotion_labels = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Contempt', 'Unknown1',
                      'Unknown2']
    predicted_emotion = emotion_labels[predicted_class[0]]

    # 显示结果
    result_label.config(text=f"Predicted Emotion: {predicted_emotion}")

# 创建GUI窗口
root = tk.Tk()
root.title("Audio Emotion Classifier")

# 添加选择文件的按钮
browse_button = tk.Button(root, text="Browse Audio File",
                          command=lambda: classify_emotion(filedialog.askopenfilename()))
browse_button.pack()

# 添加显示结果的标签
result_label = tk.Label(root, text="")
result_label.pack()

# 运行GUI
root.mainloop()