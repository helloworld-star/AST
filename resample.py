import librosa
import pyaudio
import wave
import soundfile

import wave
import pyaudio
import numpy as np
from pyaudio import PyAudio
import matplotlib.pyplot as plt


def resample_music(path):
    y, sr = librosa.load(path, sr=44100)
    y_16k = librosa.resample(y, sr, 16000)
    soundfile.write(path, y_16k, 16000)  # 重新采样的音频文件保存


def play_wav(wav_path):
    CHUNK = 1024
    # print("wav_path :",wav_path )
    #
    wf = wave.open(wav_path, 'rb')
    # print("samplewidth:", wf.getsampwidth())
    # print("channles:",wf.getnchannels())
    # print("framerate:",wf.getframerate())
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(CHUNK)
    while len(data) > 0:
        stream.write(data)
        data = wf.readframes(CHUNK)
    stream.stop_stream()
    stream.close()
    wf.close()
    p.terminate()


# 定义sound recoding 函数,其参数为录音时间t
# def sound_rec(t):
#     # 定义数据流块
#     CHUNK = 1024
#     FORMAT = pyaudio.paInt16
#     CHANNELS = 2
#     RATE = 44100
#     # 录音时间
#     RECORD_SECONDS = t
#     # 要写入的文件名
#     path_wave_load="D:/my_data/python_project/AST/sample_audios/"
#     WAVE_OUTPUT_FILENAME = path_wave_load+"output_new.wav"
#     # 创建PyAudio对象
#     p = pyaudio.PyAudio()
#     # 打开数据流
#     stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
#     print("start recording")
#     # 开始录音
#     frames = []
#     for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#         data = stream.read(CHUNK)
#         frames.append(data)
#     print("done recording")
#     # 停止数据流
#     stream.stop_stream()
#     stream.close()
#     # 关闭PyAudio
#     p.terminate()
#     # 写入录音文件
#     wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frames))
#     wf.close()
