import numpy as np
import pyaudio
import wave
import struct
import keyboard
import time
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal

# 音频录制参数设置
CHUNK = 1024 * 8  # 音频流缓冲区
FORMAT = pyaudio.paInt16  # 16位音频格式
CHANNELS = 1  # 单声道
RATE = 44100  # 采样率

WAVE_OUTPUT_FILENAME = 'signal.wav'  # 保存的音频文件名

# 创建 PyAudio 对象
p = pyaudio.PyAudio()

# 开始录制音频
print('Recording')
frames = []  # 用于存储录制的音频数据

# 创建带通滤波器
low_freq = 8000
high_freq = 10000
order = 4  # 滤波器阶数
b_band, a_band = signal.butter(order, [2 * low_freq / RATE, 2 * high_freq / RATE], btype='band')

# 创建低通滤波器
nyquist = 0.5 * RATE
cutoff = 4000  # 低通滤波器的截止频率
normal_cutoff = cutoff / nyquist
b_low, a_low = signal.butter(25, normal_cutoff, btype='low')

# 打开音频流（录制）
input_device_index = 0;
stream_in = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   input_device_index=0,
                   frames_per_buffer=CHUNK)
# 打开音频流（播放）
stream_out = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    frames_per_buffer=CHUNK)

# 创建图形
fig, axs = plt.subplots(3, figsize=(10, 12))
x = np.arange(0, CHUNK)
x_fft_noise = np.fft.fftfreq(CHUNK, 1 / RATE)[:CHUNK // 2]  # 频率轴，只取正频率部分
x_fft_cope = np.fft.fftfreq(CHUNK, 1 / RATE)[:CHUNK // 2]  # 频率轴，只取正频率部分

line, = axs[0].plot(x, np.random.rand(CHUNK), '-', lw=2)
line_fft_noise, = axs[1].plot(x_fft_noise, np.random.rand(CHUNK // 2), '-', lw=2)
line_fft_cope, = axs[2].plot(x_fft_cope, np.random.rand(CHUNK // 2), '-', lw=2)

# 格式化图形
axs[0].set_title('AUDIO WAVEFORM')
axs[0].set_xlabel('samples')
axs[0].set_ylabel('volume')
axs[0].set_ylim(0, 10000)

axs[1].set_title('Real-time Spectrum with Noise')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Amplitude')
axs[1].set_xlim(20, RATE / 2)
axs[1].set_ylim(0, 5)

axs[2].set_title('Real-time Spectrum after Filtering')
axs[2].set_xlabel('Frequency (Hz)')
axs[2].set_ylabel('Amplitude')
axs[2].set_xlim(20, RATE / 2)
axs[2].set_ylim(0, 5)

plt.ion()  # 开启交互模式
plt.show()

# 实时更新图形
while True:
    data = stream_in.read(CHUNK)
    frames.append(data)

    data_int = np.frombuffer(data, dtype=np.int16) + 5000
    line.set_ydata(data_int)

    # 去除直流（DC offset）
    data_int = data_int - np.mean(data_int)
    # 缩放音频数据，将其范围限制在合适的整幅内
    scale_factor = 1.5  # 根据需要调整缩放因子
    data_int = (data_int * scale_factor).astype(np.int16)

    # 添加低频噪声（100Hz正弦波）
    # low_freq_noise = 50 * np.sin(2 * np.pi * 100 * np.arange(CHUNK) / RATE)
    # data_int_with_noise = data_int + low_freq_noise.astype(np.int16)
    # data_int_with_noise = data_int + low_freq_noise

    # 生成带限高斯白噪声
    mean = 0  # 均值
    stddev = 10000  # 标准差，控制噪声的强度
    white_noise = np.random.normal(mean, stddev, len(data_int))
    white_noise_band = signal.lfilter(b_band, a_band, white_noise)

    # 添加高斯噪声
    # data_int_with_noise = data_int + white_noise_band.astype(np.int16)
    mixing_ratio = 0.5  # 调整混合比例
    data_int_with_noise = data_int + mixing_ratio * white_noise_band

    # FFT
    y_fft_noise = fft(data_int_with_noise)
    line_fft_noise.set_ydata(np.abs(y_fft_noise[:CHUNK // 2]) * 2 / (256 * CHUNK))

    # 低通滤波还原音频信号
    data_int_cope = signal.filtfilt(b_low, a_low, data_int_with_noise)
    data_int_cope = data_int_cope.astype(np.int16)

    # FFT
    y_fft_cope = fft(data_int_cope)
    line_fft_cope.set_ydata(np.abs(y_fft_cope[:CHUNK // 2]) * 2 / (256 * CHUNK))

    # 播放音频
    # stream_out.write(data_int.tobytes())
    # stream_out.write(data_int_with_noise.tobytes())
    stream_out.write(data_int_cope.tobytes())
    # stream_out.write(white_noise_band.tobytes())

    fig.canvas.draw()
    fig.canvas.flush_events()

    # frames.append(data)

    if keyboard.is_pressed('space'):
        print('Finished recording')
        break

# 关闭音频流和 PyAudio 对象(注：两个对象)
stream_in.stop_stream()
stream_in.close()
stream_out.stop_stream()
stream_out.close()
p.terminate()

# 将录制的音频存为wav文件
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')  # 打开文件，只写入
wf.setnchannels(CHANNELS)  # 设置音频通道数
wf.setsampwidth(p.get_sample_size(FORMAT))  # 设置以字节为单位返回样本宽度，即采样点位数
wf.setframerate(RATE)  # 设置音频采样率
wf.writeframes(b''.join(frames))  # 将音频数据写入wav文件
wf.close()  # 关闭写好的文件
