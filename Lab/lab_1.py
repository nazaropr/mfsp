import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from prettytable import PrettyTable
import warnings

warnings.filterwarnings("ignore")

beginOfInterval = -np.pi
endOfInterval = np.pi
N = 50


def f(x):
    return 15 * np.sin(15 * np.pi * x)


def bk(x, k):
    return (2 / (2 * np.pi)) * f(x) * np.sin(x * k)


def ak(x, k):
    return (2 / (2 * np.pi)) * f(x) * np.cos((k * x))


def integrateCoefficient(coefficient, k):
    return integrate.quad(coefficient, beginOfInterval, endOfInterval, args=k)[0]


def calculateHarmonic(x, k):
    return integrateCoefficient(ak, k) * np.cos((k * x)) + \
        integrateCoefficient(bk, k) * np.sin((k * x))


def calculateApproximate(x, number):
    return integrateCoefficient(ak, 0) / 2 + sum(calculateHarmonic(x, k) for k in range(1, number + 1))


def calculateError(x):
    try:
        return (calculateApproximate(x, N) - f(x)) / f(x)
    except RuntimeWarning:
        return 0


def printToFile(output_str, file_object):
    print(output_str)
    file_object.write(output_str + "\n")


def show_fourier_coefficients(file_object):
    th, td = ['№', 'b_k', 'a_k'], []
    printToFile("\nКоефіцієнти ряду Фур'є: ", file_object)
    for i in range(0, N + 1):
        td.append(i)
        if i == 0:
            td.append("-")
            td.append((round(integrateCoefficient(ak, i), 6)))
            continue
        td.append((str(round(integrateCoefficient(bk, i), 6))))
        td.append((str(round(integrateCoefficient(ak, i), 6))))
    columns = len(th)
    table = PrettyTable(th)
    while td:
        table.add_row(td[:columns])
        file_object.write('b_{0:<3} = {1:<12} a_{0:<2} = {2:<12}'.format(*td[:columns]) + "\n")
        td = td[columns:]
    print(table)


def show_graphs():
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    ax1.set_title("Графік функції f та наближення функції f рядом Фур'є", fontsize=16)
    plt.grid(True)
    x_1 = np.linspace(beginOfInterval, endOfInterval, num=1000)
    plt.plot(x_1, f(x_1), label="Функція", linewidth=3)
    plt.plot(x_1, calculateApproximate(x_1, N), label="Наближення = " + str(30), color="pink")
    plt.xlim([beginOfInterval, endOfInterval])
    plt.legend(borderaxespad=0.2, loc="best")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.set_title('Функція a(k) в частотній області', fontsize=16)
    plt.grid(True)
    for i in range(0, N + 1):
        a_value = integrateCoefficient(ak, i)
        plt.plot(i, a_value, 'ro-')
        plt.plot([i, i], [0, a_value], 'r-')
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(10, 10))
    ax3.set_title('Функція b(k) в частотній області', fontsize=16)
    plt.grid(True)
    for i in range(1, N + 1):
        b_value = integrateCoefficient(bk, i)
        plt.plot(i, b_value, 'bo-')
        plt.plot([i, i], [0, b_value], 'b-')
    plt.show()

    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 10))
    ax4.set_title("Графік функції f та поступове наближення функції f рядом Фур'є", fontsize=16)
    plt.grid(True)
    x_2 = np.linspace(beginOfInterval, endOfInterval, num=1000)
    plt.xlim([beginOfInterval, endOfInterval])
    plt.plot(x_2, f(x_2), label="Функція f")
    for i in range(1, N + 1, 4):
        plt.plot(x_2, calculateApproximate(x_2, i), label="Наближення = " + str(i))
    plt.legend(borderaxespad=0.2, loc="best")
    plt.show()

    fig5, ax5 = plt.subplots(1, 1, figsize=(10, 10))
    ax5.set_title("Графік відносної похибки апроксимації", fontsize=16)
    plt.grid(True)
    x_3 = np.linspace(beginOfInterval + 0.01, endOfInterval - 0.01, num=3500)
    plt.xlim([beginOfInterval, endOfInterval])
    plt.plot(x_3, ((calculateApproximate(x_3, N) - f(x_3)) / f(x_3)) * 100, label="Значення відносної похибки")
    plt.legend(borderaxespad=0.2, loc="best")
    plt.show()


file_object = open("./result.txt", "w", encoding="utf-8")
printToFile("Функція: 15 * sin(15 * pi * x), на проміжку х є [{}; {}]".format(beginOfInterval, endOfInterval), file_object)
show_graphs()
show_fourier_coefficients(file_object)
file_object.close()