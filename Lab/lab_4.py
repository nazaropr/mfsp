import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def generate_sequence(n, N, A, phi, a, b):
    x = np.linspace(0, b, N)
    y = exact_values(x, A, n, phi)

    max_error = 0.05 * A
    error = np.random.uniform(-max_error, max_error, N)
    y += error

    return x, y

def exact_values(x, A, n, phi):
    return A * np.sin(n * x + phi)+n

def arithmetic_mean(values):
    return np.nanmean(values)

def harmonic_mean(values):
    values = values[values != 0]  # Виключення значень, рівних 0
    if len(values) == 0:
        return 0
    else:
        return len(values) / np.nansum(1 / values)

def geometric_mean(values):
    values = values[values > 0]  # Виключення некоректних значень
    return np.exp(np.nanmean(np.log(values)))

def plot_function(x, y, label, color):
    plt.plot(x, y, label=label, color=color)
    plt.xlabel('x')
    plt.ylabel('y')

def calculate_exact_value(x, A, n, phi):
    return exact_values(x, A, n, phi)

def compare_values(approx_value, exact_value):
    absolute_error = np.abs(approx_value - exact_value)
    relative_error = np.divide(absolute_error, np.abs(exact_value), out=np.zeros_like(absolute_error), where=exact_value!=0)
    return absolute_error, relative_error

# Задані параметри
n = 15
N = 1500
A = 1.0
phi = 0.0
a = 0.0
b = np.pi/4

# Генерувати послідовність
x, y = generate_sequence(n, N, A, phi, a, b)

# Видалення некоректних значень
y = np.ma.masked_invalid(y).compressed()

# Обчислення середніх значень
arithmetic_mean_value = arithmetic_mean(y)
harmonic_mean_value = harmonic_mean(y)
geometric_mean_value = geometric_mean(y)

# Обчислення точного значення
exact_value = calculate_exact_value(x, A, n, phi)

# Порівняння максимумів та мінімумів абсолютних та відносних похибок
absolute_errors = []
relative_errors = []

approx_values = [arithmetic_mean_value, harmonic_mean_value, geometric_mean_value]
for approx_value in approx_values:
    absolute_error, relative_error = compare_values(approx_value, exact_value)
    absolute_errors.append(absolute_error)
    relative_errors.append(relative_error)

# Виведення результатів
print("\nArithmetic Mean:")
print("Arithmetic Mean:", arithmetic_mean_value)
print("Max Absolute Error:", np.nanmax(absolute_errors[0]))
print("Min Absolute Error:", np.nanmin(absolute_errors[0]))
print("Max Relative Error:", np.nanmax(relative_errors[0]))
print("Min Relative Error:", np.nanmin(relative_errors[0]))

print("\nHarmonic Mean:")
print("Harmonic Mean:", harmonic_mean_value)
print("Max Absolute Error:", np.nanmax(absolute_errors[1]))
print("Min Absolute Error:", np.nanmin(absolute_errors[1]))
print("Max Relative Error:", np.nanmax(relative_errors[1]))
print("Min Relative Error:", np.nanmin(relative_errors[1]))

print("\nGeometric Mean:")
print("Geometric Mean:", geometric_mean_value)
print("Max Absolute Error:", np.nanmax(absolute_errors[2]))
print("Min Absolute Error:", np.nanmin(absolute_errors[2]))
print("Max Relative Error:", np.nanmax(relative_errors[2]))
print("Min Relative Error:", np.nanmin(relative_errors[2]))

# Візуалізація результатів
plt.figure(figsize=(8, 6))

# Графік згенерованої послідовності
plot_function(x, y, 'Generated Sequence', 'blue')

# Графік точного значення
plot_function(x, exact_value, 'Exact Value', 'red')

plt.legend()
plt.show()
