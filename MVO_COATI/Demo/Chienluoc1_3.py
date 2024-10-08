import numpy as np
import random
import matplotlib.pyplot as plt

# Hàm fitness: Hàm rastrigin
#def rastrigin(x):
#    return 10 * len(x) + sum((xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x)

def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Khởi tạo dân số
def initialize_population(pop_size, bounds, n):
    return np.random.uniform(bounds[0], bounds[1], (pop_size, n))

# Tính toán fitness cho tất cả các vũ trụ
def calculate_fitness(population):
    return np.array([rosenbrock(ind) for ind in population])

# Tính tỷ lệ chuẩn hóa NI cho MVO
def normalize_inflation_rate(fitness):
    normalized_fitness = 1 / (fitness + 1e-6)  # Tránh chia cho 0
    return normalized_fitness / np.sum(normalized_fitness)

# Lựa chọn bằng cách quay bánh xe
def roulette_wheel_selection(weights):
    cumulative_sum = np.cumsum(weights - np.min(weights))  # Tránh giá trị âm
    normalized_weights = cumulative_sum / cumulative_sum[-1]  # Chuẩn hóa
    random_number = np.random.rand()
    index = np.searchsorted(normalized_weights, random_number)
    return index

# Cập nhật dân số bằng cơ chế của MVO
def update_population_mvo(population, fitness, best_universe, WEP, bounds, TDR):
    new_population = population.copy()
    best_fitness = np.min(fitness)

    # Tính NI
    NI = normalize_inflation_rate(fitness)
    SU = population[np.argsort(fitness)]  # Các vũ trụ được sắp xếp theo fitness

    for i in range(len(population)):
        if fitness[i] > best_fitness:  # Bỏ qua fitness tốt nhất
            r1 = np.random.rand()
            if r1 < NI[i]:  # Sử dụng NI để so sánh
                white_hole_index = roulette_wheel_selection(-NI)
                new_population[i] = SU[white_hole_index]
            else:  # Cơ chế Wormhole
                r2 = np.random.rand()
                if r2 < WEP:
                    for j in range(len(population[i])):
                        r3 = np.random.rand()
                        r4 = np.random.rand()
                        if r3 < 0.5:
                            new_population[i][j] = best_universe[j] + TDR * ((bounds[1] - bounds[0]) * r4 + bounds[0])
                        else:
                            new_population[i][j] = best_universe[j] - TDR * ((bounds[1] - bounds[0]) * r4 + bounds[0])
                else:
                    new_population[i] = population[i]

    return new_population

# Cập nhật vị trí theo COA
def update_population_coa(population, best_solution, bounds):
    N, m = population.shape
    r = random.random()
    I = random.randint(1, 2)
    
    new_population = population.copy()

    # Iguana là cá thể tốt nhất (best solution)
    iguana = best_solution.copy()

    for i in range(N):
        if i >= N / 2:
            # Coati rơi xuống đất và di chuyển để bắt Iguana
            iguana_G = np.random.uniform(bounds[0], bounds[1], m)
            if rosenbrock(iguana_G) < rosenbrock(population[i]):
                new_population[i] = population[i] + r * (iguana_G - I * population[i])
            else:
                new_population[i] = population[i] + r * (population[i] - iguana_G)
        else:
            # Coati trên cây đe dọa Iguana
            new_population[i] = population[i] + r * (iguana - I * population[i])
        
        # Áp dụng yếu tố nhiễu
        noise = np.random.uniform(-0.001, 0.001, m)
        new_population[i] += noise

    return new_population

# Thuật toán lai COA-MVO
def hybrid_coa_mvo(pop_size, bounds, max_iter, n, switch_iter):
    # Khởi tạo dân số và tính fitness ban đầu
    population = initialize_population(pop_size, bounds, n)
    fitness = calculate_fitness(population)
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    WEPmin, WEPmax = 0.2, 0.8
    TDR = 1.0

    for iter in range(max_iter):
        WEP = WEPmin + (iter / max_iter) * (WEPmax - WEPmin)
        TDR = 1 - (iter / max_iter)

        # Cập nhật dân số theo chiến lược kết hợp
        if iter < switch_iter:
            # Giai đoạn đầu: chủ yếu MVO
            population = update_population_mvo(population, fitness, best_solution, WEP, bounds, TDR)
        else:
            # Giai đoạn sau: chuyển sang COA
            population = update_population_coa(population, best_solution, bounds)
        
        # Tính toán lại fitness
        fitness = calculate_fitness(population)

        # Cập nhật cá thể tốt nhất
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[current_best_idx].copy()

        print(f"Iteration {iter + 1}: Best Fitness = {best_fitness:.6f}, Best Solution = {best_solution}")

        # Hiển thị kết quả
        plt.clf()  # Xóa biểu đồ hiện tại
        plt.scatter(population[:, 0], population[:, 1], color='blue', alpha=0.5)
        plt.xlim(bounds[0], bounds[1])
        plt.ylim(bounds[0], bounds[1])
        plt.title(f'Iteration {iter + 1}')
        plt.xlabel('Variable 1')
        plt.ylabel('Variable 2')

        plt.axhline(0, color='k', lw=1, ls='--')  # Đường ngang tại y=0
        plt.axvline(0, color='k', lw=1, ls='--')  # Đường dọc tại x=0

        plt.grid()
        plt.pause(0.01)

    return best_solution, best_fitness

# Tham số cho thuật toán
pop_size = 2000
bounds = [-5, 5]  # Giới hạn cho hàm rosenbrock
max_iter = 1000
n = 4
switch_iter = 500  # Chuyển từ MVO sang COA sau 500 vòng lặp

# Chạy thuật toán
best_solution, best_fitness = hybrid_coa_mvo(pop_size, bounds, max_iter, n, switch_iter)

# Kết quả tốt nhất
print("Best solution found:")
for idx, value in enumerate(best_solution):
    print(f"  Variable {idx + 1}: {value:.6f}")
print(f"With fitness: {best_fitness:.6f}")

plt.show()
