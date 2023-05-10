import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

n = 15
N = 10 + n

num_add = num_mult = N

# Обчислення k-го члена ряду Фур'є
def fourier_coefficient_k(k, x):
    global num_mult, num_add
    N = len(x)
    n = np.arange(N)
    c = np.exp(-2j * np.pi * k * n / N)
    num_mult += 4 * N
    num_add += N - 1
    return np.dot(x, c)


# обчислення коефіцієнта Фур'є C_k
def fourier_coefficient(k, x):
    A_k = fourier_coefficient_k(k, x.real)
    B_k = fourier_coefficient_k(k, x.imag)
    return A_k + 1j * B_k


# Генерація довільного вектора f
f = np.random.rand(N) + 1j * np.random.rand(N)

# Обчислення коефіцієнтів Фур'є та часу обчислення
start_time = time.time()
C = [fourier_coefficient(k, f) for k in range(N)]
end_time = time.time()
elapsed_time = end_time - start_time
print("\nЧас обчислення: ", elapsed_time)

# Обчислення спектру амплітуд та фаз та побудова графіків
amp_spectrum = np.abs(C)
phase_spectrum = np.angle(C)
freq_axis = np.arange(N)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.stem(freq_axis, amp_spectrum, 'blue')
plt.title('Амплітудний спектр')
plt.xlabel('Частота')
plt.ylabel('Амплітуда')

plt.subplot(1, 2, 2)
plt.stem(freq_axis, phase_spectrum, 'blue')
plt.title('Фазовий спектр')
plt.xlabel('Частота')
plt.ylabel('Фаза')
plt.show()


print("\nКількість операцій множення: ", num_mult)
print("Кількість операцій додавання: ", num_add, "\n")

for i, c in enumerate(C):
    print(f'C_{i} = {c}')