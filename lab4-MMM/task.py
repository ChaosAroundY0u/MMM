import numpy as np
import matplotlib.pyplot as plt
"""------------------------------------#1-----------------------------------"""
# data = np.fromfile("var01_z1.bin", dtype = np.int16)
# print(data)
# t_axe = len(data)
# print(t_axe)
# data = data[32575:] #cutting the list because of rapid growth of graph
# sr = 1000

# def running_avg(mass):
#     ans = []
#     moving_sum = 0
#     for elem in mass[:10000]:
#         moving_sum += elem
#     ans.append(moving_sum / 10000)
#     for i in range(len(mass) - 10000):
#         moving_sum += (mass[i + 10000] - mass[i])
#         ans.append(moving_sum / 10000)
#     return ans


# plt.plot(data)
# plt.plot(running_avg(data), color = "red")
"""------------------------------------#2-----------------------------------"""
# data = np.fromfile("var01_z2.bin", dtype = np.float64)
# print(data)
# print(len(data))
# sr = 1000
# f_axe = np.fft.fftfreq(len(data), sr)
# #plt.plot(data)
# #plt.plot(f_axe,color = "red")

# def sin_window(elem, mass):
#     return np.sin(np.pi * elem / (len(mass) - 1))

# def multiply(mass):
#     ans = []
#     for i in range(len(mass)):
#         ans.append(mass[i]*sin_window(i, mass))
#     return ans

# spec = np.fft.fftshift(np.fft.fft(multiply(data)))
# f_axe = np.fft.fftshift(f_axe)
# plt.plot(f_axe, 20*np.log10(spec))
# plt.grid(True)
"""-------------------------------#3-mannualy-------------------------------"""
data = np.fromfile("var01_z3.bin", dtype = np.float64)
print(data)
sr = 1000
f_axe = np.fft.fftfreq(len(data), sr)
window_length = 1000
overlap = 999

spectrogram = []
for i in range(0, len(data) - window_length, overlap):
    window = np.sin(np.pi * np.arange(1000)/ (len(data) - 1)) * data[i:i+window_length]
    spectr = np.abs(np.fft.fft(window))[:window_length//2]
    spectrogram.append(spectr)


plt.imshow(20*np.log10(np.transpose(np.array(spectrogram))),aspect='auto', origin='lower')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.colorbar(label = 'dB')


# overlap = 999
# # spec = []
# # for i in range(0, len(data)):
# #     window = np.sin(np.pi * np.arange(1000)/ (len(data) - 1)) *  data[i:i+1000]
# #     spectr = np.fft.fft(window)[:1000]
# #     spec.append(spectr)

# def sin_window(elem, mass):
#      return np.sin(np.pi * elem / (len(mass) - 1))
    
# def spectrogram(mass):
#     ans = []
#     for i in range(len(mass)):
#         window = [sin_window(elem, mass) for elem in mass[i:i+1000]]
#         ans.append(np.abs(np.fft.fft(window)))
#     return ans
# print(spectrogram(data))


"""-----------------------------------#3------------------------------------"""
# from scipy.signal import spectrogram
# data = np.fromfile("var01_z3.bin", dtype = np.float64)
# print(data)
# print(len(data))
# sr = 1000
# # f_axe = np.fft.fftfreq(len(data), sr)
    
# # def sin_window(elem, mass):
# #     return np.sin(np.pi * elem / (len(mass) - 1))

# # def multiply(mass):
# #     ans = []
# #     for i in range(len(mass)):
# #         ans.append(mass[i]*sin_window(i, mass))
# #     return ans

# f, t, Sxx = spectrogram(data, sr, window = np.sin(np.pi * np.arange(1000)/ (len(data) - 1)), noverlap = 999)
# # f_axe = np.fft.fftshift(f_axe)
# plt.pcolormesh(t, f, np.log(Sxx))
