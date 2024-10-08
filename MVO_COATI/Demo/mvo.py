import numpy as np
import matplotlib.pyplot as plt

# Hàm fitness: Hàm Rastrigin
def rastrigin(x):
    return 10 * len(x) + sum((xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x)

# Khởi tạo dân số
def initialize_population(pop_size, bounds, n):
    return np.random.uniform(bounds[0], bounds[1], (pop_size, n))

# Tính toán fitness cho tất cả các vũ trụ
def calculate_fitness(population):
    return np.array([rastrigin(ind) for ind in population])

# Tính tỷ lệ chuẩn hóa NI
def normalize_inflation_rate(fitness):
    normalized_fitness = 1 / (fitness + 1e-6)  # Tránh chia cho 0
    return normalized_fitness / np.sum(normalized_fitness)

# Cập nhật Whitehole/Blackhole
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
                    # Nếu không thỏa mãn WEP, giữ nguyên giá trị cũ
                    new_population[i] = population[i]

    return new_population

# Lựa chọn bằng cách quay bánh xe
def roulette_wheel_selection(weights):
    cumulative_sum = np.cumsum(weights - np.min(weights))  # Tránh giá trị âm
    normalized_weights = cumulative_sum / cumulative_sum[-1]  # Chuẩn hóa
    random_number = np.random.rand()
    index = np.searchsorted(normalized_weights, random_number)
    return index

# MVO algorithm
def mvo(pop_size, bounds, max_iter, n, stagnation_limit=100):
    population = initialize_population(pop_size, bounds, n)
    fitness = calculate_fitness(population)
    best_idx = np.argmin(fitness)
    best_universe = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    WEPmin, WEPmax = 0.2, 0.8
    TDR = 1.0

    stagnation_counter = 0  # Biến đếm cho số vòng lặp không thay đổi fitness

    for iter in range(max_iter):
        WEP = WEPmin + (iter / max_iter) * (WEPmax - WEPmin)
        TDR = 1 - (iter / max_iter)

        # Cập nhật dân số
        population = update_population_mvo(population, fitness, best_universe, WEP, bounds, TDR)
        fitness = calculate_fitness(population)

        # Cập nhật fitness và vũ trụ tốt nhất
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_universe = population[current_best_idx].copy()
            stagnation_counter = 0  # Reset đếm khi có sự cải thiện
        else:
            stagnation_counter += 1  # Tăng đếm nếu không có cải thiện

        print(f"Iteration {iter + 1}: Best Fitness = {best_fitness:.6f}, Best Solution = {best_universe}")

        # Kiểm tra điều kiện dừng
        if stagnation_counter >= stagnation_limit:
            print(f"Stopping criteria met at iteration {iter + 1}.")
            break

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

    return best_universe, best_fitness

# Tham số
pop_size = 100 
bounds = [-2, 2]  # Giới hạn cho hàm Rastrigin
max_iter = 1000
n = 4

best_solution, best_fitness = mvo(pop_size, bounds, max_iter, n)
print("Best solution found:")
for idx, value in enumerate(best_solution):
    print(f"  Variable {idx + 1}: {value:.6f}")
print(f"With fitness: {best_fitness:.6f}")

plt.show()
