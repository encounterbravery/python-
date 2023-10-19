import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import convolve
import soundfile as sf
import matplotlib.pyplot as plt

# 读取音频文件
filename = '是风动.wav'
x, fs = wav.read(filename)

# 信源编码
x_normalized = x / max(abs(x))
bits = 8
quantized_int = np.round((2**bits - 1) * (x_normalized + 1) / 2).astype(int)
binary_code = np.unpackbits(quantized_int.astype(np.uint8))

# 信道编码处理
K = 3
n = 2**K - 1
G = np.array([[1, 0, 1, 1], [1, 1, 0, 1]])
M = 16
trellis = [5, 7]
conv_encoder = convolve(binary_code, G, mode='full')
conv_encoder = np.mod(conv_encoder, M)

# 传输过程（简单模拟加高斯白噪声）
snr = 10
noise_power = 10**(-snr / 10)
noisy_coded = conv_encoder + np.sqrt(noise_power / 2) * np.random.randn(len(conv_encoder))

# 信道解码处理
vitdec_decoder = [5, 7]
decoded = np.zeros_like(noisy_coded, dtype=int)
decoded[:len(decoded) - (n - 1)] = np.unpackbits(noisy_coded[:len(decoded) - (n - 1)].astype(np.uint8))
decoded = np.packbits(decoded)
decoded = decoded[:len(decoded) - (n - 1)]

# 信源解码
dequantized_int = np.reshape(decoded, (-1, bits))
dequantized = (2 * dequantized_int / (2**bits - 1)) - 1

# 恢复幅值
x_restored = dequantized * np.max(np.abs(x))

# 将信号写入新的音频文件
sf.write('output1.wav', -1/2 * x_restored, fs)

# 读取原始信号和解码后信号
x = sf.read('是风动.wav')[0]
y = sf.read('output1.wav')[0]

# 计算均方根误差（RMSE）
rmse = np.sqrt(np.mean((x - y)**2))
print(f"原始信号与解码后信号的均方根误差为：{rmse}")

# 画出时域波形图
t = np.arange(0, len(x)) / fs
plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.xlabel('时间（秒）')
plt.ylabel('幅度')
plt.title('原始信号')

plt.subplot(2, 1, 2)
plt.plot(t, y)
plt.xlabel('时间（秒）')
plt.ylabel('幅度')
plt.title('解码后信号')

# 调整窗口大小
fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.show()
