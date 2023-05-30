import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

def fft(x):
    n = x.shape[0]
    if n == 1:
        return x

    even = fft(x[::2])
    odd = fft(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(n) / n)
    global multiplication_count, addition_count
    multiplication_count += n // 2
    addition_count += n - 1
    return np.concatenate([even + factor[:n // 2] * odd,
                           even + factor[n // 2:] * odd])

def fourier_coefficient(f, k):
    N = len(f)
    n = np.arange(N)
    Ak = np.sum(f * np.cos(2 * np.pi * k * n / N))
    Bk = np.sum(f * np.sin(2 * np.pi * k * n / N))
    Ck = Ak - 1j * Bk
    num_multiplications = 8 * N + 1
    num_additions = 2 * (N - 1)
    #print("С_" + str(k) + " = " + str(Ck))
    return Ck, num_multiplications, num_additions

def fourier_coefficients(f):
    N = len(f)
    C = np.zeros(N, dtype=complex)
    total_multiplications = 0
    total_additions = 0
    for k in range(N):
        C[k], num_multiplications, num_additions = fourier_coefficient(f, k)
        total_multiplications += num_multiplications
        total_additions += num_additions
    return C, total_multiplications, total_additions

N = 20
x = np.random.rand(N)
multiplication_count = 0
addition_count = 0

M = 2**int(np.ceil(np.log2(N)))
x = np.concatenate([x, np.zeros(M-N)])
#print("Вхідний вектор = ", x)

start_time = time.time()
C, total_multiplications, total_additions = fourier_coefficients(x)
end_time = time.time()
execution_time = end_time - start_time
print("\nРезультат ДПФ = ", C)
print("Час виконання ДПФ: ", execution_time, " секунд")
print("Кількість операцій множення - ", total_multiplications)
print("Кількість операцій додавання - ",total_additions)

start_time = time.time()
y = fft(x)
end_time = time.time()

addition_count = N
multiplication_count = 4 * N

execution_time = end_time - start_time
print("\nРезультат ШПФ = ", y)
print("Час виконання ШПФ: ", execution_time, " секунд")
print("Кількість операцій множення - ", multiplication_count)
print("Кількість операцій додавання - ",addition_count)
print("Рівність із бібліотечною функцією - ", np.allclose(y, np.fft.fft(x)))


amplitude_spectrum = np.abs(y)
phase_spectrum = np.angle(y)

plt.plot
plt.stem(amplitude_spectrum, 'green')
plt.title("Спектр амплітуд")
plt.show()

plt.plot
plt.stem(phase_spectrum, 'green')
plt.title("Спектр фаз")
plt.show()