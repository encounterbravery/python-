import wave
import pyaudio
import keyboard
import numpy as np
import matplotlib.pyplot as plt

# 打开wav文件
wf = wave.open('是风动.wav', 'rb')
# 获取wav文件的参数
sample_width = wf.getsampwidth()  # 每个样本的字节宽度（以字节为单位）
frameRate = wf.getframerate()  # 采样率
nChannels = wf.getnchannels()  # 声道数（1表示单声道，2表示立体声）
# audio_data = wf.readframes(-1)  # 读取整个音频数据
# print(sample_width, ' ', frameRate, ' ', nChannels)

# 创建 PyAudio 对象
p = pyaudio.PyAudio()

# 初始化音频流播放
stream = p.open(format=p.get_format_from_width(sample_width),
                channels=nChannels,
                rate=frameRate,
                output=True)

# 创建图形
CHUNK = 1024 * 8   # 音频流缓冲区
fig, axs = plt.subplots(3, figsize=(10, 12))
x = np.arange(0, CHUNK)

line_initial, = axs[0].plot(x, np.random.rand(CHUNK), '-', lw=2)
line_decode, = axs[1].plot(x, np.random.rand(CHUNK), '-', lw=2)
line_error, = axs[2].plot(x, np.random.rand(CHUNK), '-', lw=2)

# 格式化图形
axs[0].set_title('Audio before Coding')  # 编码前的原始信号
axs[0].set_xlabel('samples')
axs[0].set_ylabel('volume')
axs[0].set_ylim(0, 30000)

axs[1].set_title('Audio after Decoding')  # 解码后的信号
axs[1].set_xlabel('samples')
axs[1].set_ylabel('volume')
axs[1].set_ylim(0, 10000)

axs[2].set_title('Audio Error')  # 编码前和解码后信号之间的误差
axs[2].set_xlabel('samples')
axs[2].set_ylabel('volume')
axs[2].set_ylim(0, 10000)

plt.ion()  # 开启交互模式
plt.show()

# 播放音频,并实时更新图像
print('Playing')
while True:
    audio_data = wf.readframes(CHUNK)
    stream.write(audio_data)
    if not audio_data:
        break  # 文件读取结束自动退出程序

    # 将音频数据转换为NumPy数组
    audio_data_int = np.frombuffer(audio_data, dtype=np.int16) + 15000
    print(audio_data)
    print(audio_data_int)
    line_initial.set_ydata(audio_data_int)

    plt.pause(0.00001)

    # fig.canvas.draw()
    # fig.canvas.flush_events()

    # 空格键退出循环
    if keyboard.is_pressed('space'):
        break

print('Finished Playing')

# 关闭音频流
stream.stop_stream()
stream.close()

# 关闭PyAudio对象
p.terminate()

# 关闭wav文件
wf.close()
 