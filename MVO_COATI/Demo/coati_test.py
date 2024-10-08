import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA

# Nhập thông tin đầu vào
N = int(input('Nhập số lượng Coati: '))
T = int(input('Nhập số vòng lặp: '))
m = int(input('Nhập số chiều không gian (n): '))
lbj = -5
ubj = 5

# Khởi tạo mảng 2 chiều với N Coati và m biến quyết định
x = np.random.uniform(lbj, ubj, (N, m))

# Khởi tạo giá trị fitness cho quần thể ban đầu
fitness_values = np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2, axis=1)

# Khởi tạo danh sách để lưu trữ lịch sử vị trí
search_history = [[] for _ in range(T)]
best_fitness_history = []
best_x_history = []

# Tìm cá thể có fitness tốt nhất ban đầu
best_fitness = np.min(fitness_values)
best_x = x[np.argmin(fitness_values)].copy()
global_best_fitness = best_fitness
global_best_x = best_x.copy()

# Bắt đầu vòng lặp thuật toán
for t in range(T):
    print(f"\nLần {t + 1}:")

    r = random.random()
    I = random.randint(1, 2)

    # Cập nhập vị trí của Iguana (cá thể tốt nhất)
    iguana = best_x
    iguana_fitness = best_fitness
    print(f"Iguana = {iguana}, Fitness = {iguana_fitness}")

    for i in range(N):
        search_history[t].append(x[i].copy())

        # Iguana rơi xuống đất và Coati di chuyển để bắt Iguana
        if i >= N / 2:
            # Vị trí rơi của Iguana:
            iguana_G = np.random.uniform(lbj, ubj, m)
            fitness_G = np.sum(100 * (iguana_G[1:m] - iguana_G[0:m-1] ** 2) ** 2 + (iguana_G[0:m-1] - 1) ** 2)
            if fitness_G < fitness_values[i]:
                x[i] = x[i] + r * (iguana_G - I * x[i])
            else:
                x[i] = x[i] + r * (x[i] - iguana_G)

        # Vị trí của Coati trên cây
        else:
            # Coati trên cây đe dọa Iguana
            x[i] = x[i] + r * (iguana - I * x[i])
        # Áp dụng yếu tố nhiễu
        noise = np.random.uniform(-0.1, 0.1, m)
        x[i] += noise

    # Tính lại giá trị fitness
    fitness_values = np.sum(100 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (x[:, :-1] - 1) ** 2, axis=1)

    # Phase 2: Escaping strategy
    lbj_local = lbj
    ubj_local = ubj
    x += +(1 - 2 * r) * (lbj_local + r * (ubj_local - lbj_local))

    # Thêm nhiễu sau khi di chuyển
    noise = np.random.uniform(-0.1, 0.1, (N, m))
    x += noise

    # Tính lại fitness sau khi di chuyển
    fitness_values = np.sum(x[:, :2] ** 2, axis=1) + np.abs(x[:, 2])

    # Cập nhật cá thể có fitness tốt nhất trong vòng lặp
    current_best_fitness = np.min(fitness_values)
    current_best_x = x[np.argmin(fitness_values)].copy()
    print(f'The best solution: {current_best_x}, Fitness = {current_best_fitness}')

    if current_best_fitness < global_best_fitness:
        global_best_fitness = round(current_best_fitness, 6)
        global_best_x = current_best_x.copy()

    best_fitness_history.append(global_best_fitness)
    best_x_history.append(global_best_x.copy())

print(f"\n=>The best solution in COA: {global_best_x}, Fitness = {global_best_fitness}")

# Áp dụng PCA để giảm số chiều xuống 2D cho toàn bộ dữ liệu
all_search_history = [point for sublist in search_history for point in sublist]
pca = PCA(n_components=2)
all_search_history_2d = pca.fit_transform(np.array(all_search_history))

# Phân chia dữ liệu đã giảm chiều thành từng vòng lặp
search_history_2d = []
start_idx = 0
for t in range(T):
    end_idx = start_idx + len(search_history[t])
    search_history_2d.append(all_search_history_2d[start_idx:end_idx])
    start_idx = end_idx

# Tạo animation của lịch sử vị trí
fig, ax = plt.subplots(figsize=(8, 6))

def update(frame):
    ax.clear()
    if frame >= len(search_history_2d):
        return
    search_history_j1 = search_history_2d[frame][:, 0]
    search_history_j2 = search_history_2d[frame][:, 1]

    # Vẽ tất cả các điểm của quần thể tại vòng lặp hiện tại
    ax.scatter(search_history_j1, search_history_j2, c='red', marker='o', s=50, alpha=0.75)

    # Vẽ điểm có fitness nhỏ nhất
    if frame < len(best_x_history):
        best_pos_2d = pca.transform([best_x_history[frame]])[0]
        best_fitness = best_fitness_history[frame]
        ax.scatter(best_pos_2d[0], best_pos_2d[1], c='green', marker='o', s=50)
        ax.text(best_pos_2d[0], best_pos_2d[1], f'Fitness = {best_fitness}', fontsize=12, ha='right')

    # Hiển thị điểm có giá trị fitness nhỏ nhất của vòng lặp hiện tại
    ax.text(0.05, 0.95, f'The best solution: {best_x_history[frame]}, Fitness = {best_fitness}', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    ax.set_title(f'Lịch sử tìm kiếm của các Coati - Lần {frame + 1}')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.grid(True)

ani = FuncAnimation(fig, update, frames=T, repeat=False)
plt.show()
