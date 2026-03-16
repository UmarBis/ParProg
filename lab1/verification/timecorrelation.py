import matplotlib.pyplot as plt

N = [50, 100, 200, 500]
t = [0.004, 0.01, 0.07, 1.02]
plt.plot(N, t)
plt.xlabel("Размер N")
plt.ylabel("Время выполнения (с)")
plt.title("Корреляция времени выполнения от размера матрицы")
plt.grid()
plt.show()
