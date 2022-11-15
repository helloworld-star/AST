import os
import sys
import librosa
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtWidgets import QMainWindow
from pyqt5_gui import Ui_Form  # 加载我们的布局
import wave
import pyaudio
import torch, torchaudio, timm
import numpy as np
from torch.cuda.amp import autocast
from src.models import ASTModel
import soundfile
from detect import ASTModelVis, make_features, load_label
from matplotlib import pyplot as plt


def resample_music(path):
    y, sr = librosa.load(path, sr=44100)
    y_16k = librosa.resample(y, sr, 16000)
    soundfile.write(path, y_16k, 16000)  # 重新采样的音频文件保存


class UsingTest(QMainWindow, Ui_Form):
    def __init__(self, *args, **kwargs):
        super(UsingTest, self).__init__(*args, **kwargs)
        self.setupUi(self)  # 初始化ui
        # 在这里，可以做一些UI的操作了，或者是点击事件或者是别的
        # 选择文件
        self.File.clicked.connect(self.select_file)
        # 设置录音时间
        self.Record.clicked.connect(self.sound_rec)
        self.Play.clicked.connect(self.play_wav)
        # 测试
        self.Detect_2.clicked.connect(self.detect_music)
        self.filename = "sample_audios/1-5996-A-6.wav"

    def select_file(self):
        self.filename, _ = QFileDialog.getOpenFileName(self, "getOpenFileName",
                                                       './sample_audios',  # 文件的起始路径
                                                       "WAV (*.wav);;JSON Files (*.json)")  # 设置文件类型
        self.print_filename.setText(self.filename)
        print(self.filename)

    def play_wav(self):
        CHUNK = 1024
        # print("wav_path :",wav_path )
        wav_path = self.filename
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
        print('Listen to this sample: ')

    def sound_rec(self):
        # 定义数据流块
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        # 录音时间
        RECORD_SECONDS = 10
        # 要写入的文件名
        path_wave_load = "sample_audios/"
        WAVE_OUTPUT_FILENAME = path_wave_load + "output_new.wav"
        self.filename = WAVE_OUTPUT_FILENAME
        print("录音")
        print(self.filename)
        # 创建PyAudio对象
        p = pyaudio.PyAudio()
        # 打开数据流
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("start recording")
        # 开始录音
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("done recording")
        # 停止数据流
        stream.stop_stream()
        stream.close()
        # 关闭PyAudio
        p.terminate()
        # 写入录音文件
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    def detect_music(self):
        global audio_model
        input_tdim = 1024
        checkpoint_path = 'pretrained_models/audio_mdl.pth'
        label_csv = './egs/audioset/data/class_labels_indices.csv'
        # music_path = "D:/my_data/python_project/dsp_project/audio/TAU-urban-acoustic-scenes-2020-mobile-development" \
        #              "/audio/public_square-barcelona-108-3100-s3.wav"
        music_path = self.filename
        resample_music(music_path)
        ast_mdl = ASTModelVis(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False,
                              audioset_pretrain=False)
        print(f'[*INFO] load checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
        audio_model.load_state_dict(checkpoint)
        audio_model = audio_model.to(torch.device("cpu"))
        audio_model.eval()
        labels = load_label(label_csv)
        feats = make_features(music_path, mel_bins=128)  # shape(1024, 128)
        global feats_data
        feats_data = feats.expand(1, input_tdim, 128)  # reshape the feature
        feats_data = feats_data.to(torch.device("cpu"))
        # Make the prediction
        with torch.no_grad():
            with autocast():
                output = audio_model.forward(feats_data)
                output = torch.sigmoid(output)
        result_output = output.data.cpu().numpy()[0]
        sorted_indexes = np.argsort(result_output)[::-1]
        print("Detect")
        print(self.filename)
        # Print audio tagging top probabilities
        # print('Predice results:')
        # for k in range(10):
        #     print('- {}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))
        # 设置输出
        # np.array(labels)[sorted_indexes[k]].tostring()
        self.Detect_Name.setText(
            str((labels)[sorted_indexes[0]]) + ':' + str(result_output[sorted_indexes[0]]) + '\n' +
            str((labels)[sorted_indexes[1]]) + ':' + str(result_output[sorted_indexes[1]]) + '\n' +
            str((labels)[sorted_indexes[2]]) + ':' + str(result_output[sorted_indexes[2]]) + '\n' +
            str((labels)[sorted_indexes[3]]) + ':' + str(result_output[sorted_indexes[3]]) + '\n' +
            str((labels)[sorted_indexes[4]]) + ':' + str(result_output[sorted_indexes[4]]) + '\n' +
            str((labels)[sorted_indexes[5]]) + ':' + str(result_output[sorted_indexes[5]]) + '\n' +
            str((labels)[sorted_indexes[6]]) + ':' + str(result_output[sorted_indexes[6]]) + '\n' +
            str((labels)[sorted_indexes[7]]) + ':' + str(result_output[sorted_indexes[7]]) + '\n' +
            str((labels)[sorted_indexes[8]]) + ':' + str(result_output[sorted_indexes[8]]) + '\n' +
            str((labels)[sorted_indexes[9]]) + ':' + str(result_output[sorted_indexes[9]])
        )

        self.original_plt()
        image = QImage('img/Original_Spectrogram.jpg')
        self.Origin.setPixmap(QPixmap.fromImage(image))
        self.resize(image.width(), image.height())
        self.predict()
        image_mult = QImage('img/Mean_Attention_Map_of_Layer.jpg')
        self.Layer.setPixmap(QPixmap.fromImage(image_mult))
        self.resize(image_mult.width(), image_mult.height())

    def original_plt(self):
        plt.figure()
        plt.title('Original Spectrogram')
        plt.imshow(feats_data[0].t().cpu(), origin='lower')
        plt.savefig('./img/Original_Spectrogram.jpg')

    def predict(self):
        with torch.no_grad():
            with autocast():
                att_list = audio_model.module.forward_visualization(feats_data)
        plt.figure()
        for i in range(len(att_list)):
            att_list[i] = att_list[i].data.cpu().numpy()
            att_list[i] = np.mean(att_list[i][0], axis=0)
            att_list[i] = np.mean(att_list[i][0:2], axis=0)
            att_list[i] = att_list[i][2:].reshape(12, 101)
            plt.subplot(6, 2, i + 1)
            plt.imshow(att_list[i], origin='lower')
            plt.title('Mean Attention Map of Layer #{:d}'.format(i))
        plt.savefig('./img/Mean_Attention_Map_of_Layer.jpg')


if __name__ == '__main__':  # 程序的入口
    app = QApplication(sys.argv)
    win = UsingTest()
    win.show()
    sys.exit(app.exec_())
