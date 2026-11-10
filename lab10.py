import math
import matplotlib.pyplot as plt

# ============================================================
# ЗАДАЧА КОШІ
# ============================================================
def f(x, y):
    return y - x

def exact_solution(x):
    return math.exp(x) + x + 1

# ============================================================
# ЧАСТИНА 2: МЕТОД РУНГЕ-КУТТА 4-ГО ПОРЯДКУ (RK4)
# ============================================================
def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h, y + h*k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def solve_rk4_fixed(a, b, y0, h):
    x_vals = [a]
    y_vals = [y0]
    x = a
    y = y0
    while x < b - 1e-9:
        y = rk4_step(x, y, h)
        x += h
        x_vals.append(x)
        y_vals.append(y)
    return x_vals, y_vals

def rk4_local_error_runge(x, y, h):
    y_full = rk4_step(x, y, h)
    y_half_1 = rk4_step(x, y, h/2)
    y_half_2 = rk4_step(x + h/2, y_half_1, h/2)
    return (16.0 / 15.0) * abs(y_half_2 - y_full)

# ============================================================
# ЧАСТИНА 1: МЕТОД АДАМСА 2-ГО ПОРЯДКУ (ПРОГНОЗ-КОРЕКЦІЯ)
# ============================================================
def adams_pc_2nd(x, y_prev, y, h):
    f_prev = f(x - h, y_prev)
    f_curr = f(x, y)
    y_pred = y + (h/2) * (3*f_curr - f_prev)
    x_next = x + h
    f_pred = f(x_next, y_pred)
    y_corr = y + (h/2) * (f_pred + f_curr)
    error_est = (5.0 / 6.0) * abs(y_corr - y_pred)
    return x_next, y_corr, y_pred, error_est

def solve_adams_fixed(a, b, y0, h):
    x_vals = [a]
    y_vals = [y0]
    x = a
    y = y0
    
    # Перший крок RK4
    y1 = rk4_step(x, y, h)
    x_vals.append(x + h)
    y_vals.append(y1)
    
    errors_exact = [abs(y0 - exact_solution(a)), abs(y1 - exact_solution(x + h))]
    errors_est = [0, 0]
    
    x_prev, y_prev = x, y
    x_curr, y_curr = x + h, y1
    
    while x_curr < b - 1e-9:
        x_next, y_next, y_pred, err = adams_pc_2nd(x_curr, y_prev, y_curr, h)
        errors_exact.append(abs(y_next - exact_solution(x_next)))
        errors_est.append(err)
        x_vals.append(x_next)
        y_vals.append(y_next)
        x_prev, y_prev = x_curr, y_curr
        x_curr, y_curr = x_next, y_next
        
    return x_vals, y_vals, errors_exact, errors_est

# ============================================================
# АДАПТИВНИЙ ВИБІР КРОКУ (ВИПРАВЛЕНО)
# ============================================================
def solve_adaptive(a, b, y0, h_init, eps):
    x = a
    y = y0
    h = h_init
    h_max = 0.1 # Обмеження, щоб графік був плавним
    h_min = 1e-6
    
    h_history = []
    x_history = []
    
    while x < b - 1e-9:
        if x + h > b:
            h = b - x
        
        err = rk4_local_error_runge(x, y, h)
        y_new = rk4_step(x, y, h)
        
        # Логіка зміни кроку
        if err > eps and h > h_min:
            h = h / 2.0
            continue
        elif err < eps / 10.0 and h < h_max:
            h = h * 1.5 # Плавно збільшуємо
            
        x_history.append(x)
        h_history.append(h)
        x += h
        y = y_new
        
    return x_history, h_history

# ============================================================
# ГОЛОВНА ПРОГРАМА
# ============================================================
def main():
    print("=" * 70)
    print("ЛАБОРАТОРНА РОБОТА №10")
    print("Звичайні диференціальні рівняння. Методи Рунге-Кутта та Адамса")
    print("=" * 70)
    
    a, b = 0.0, 2.0
    y0 = 2.0
    h_fixed = 0.01
    eps = 1e-5

    # --- ЧАСТИНА 1: МЕТОД АДАМСА ---
    print("\nЧАСТИНА 1: Метод прогнозу та корекції Адамса (2-й порядок)")
    x_ad, y_ad, err_ex_ad, err_est_ad = solve_adams_fixed(a, b, y0, h_fixed)
    print(f"Крок: h = {h_fixed}")
    print(f"Максимальна похибка (точно): {max(err_ex_ad):.6e}")
    print(f"Максимальна оцінка похибки: {max(err_est_ad):.6e}")

    # --- ЧАСТИНА 2: МЕТОД РУНГЕ-КУТТА ---
    print("\nЧАСТИНА 2: Метод Рунге-Кутта (4-й порядок)")
    x_rk, y_rk = solve_rk4_fixed(a, b, y0, h_fixed)
    err_ex_rk = [abs(y_rk[i] - exact_solution(x_rk[i])) for i in range(len(x_rk))]
    err_est_rk = [0.0]
    for i in range(len(x_rk)-1):
        err_est_rk.append(rk4_local_error_runge(x_rk[i], y_rk[i], h_fixed))
        
    print(f"Крок: h = {h_fixed}")
    print(f"Максимальна похибка (точно): {max(err_ex_rk):.6e}")
    print(f"Максимальна оцінка похибки (Рунге): {max(err_est_rk):.6e}")

    # --- АДАПТИВНИЙ КРОК ---
    print("\nАдаптивний вибір кроку (eps = 1e-5):")
    x_hist_rk, h_hist_rk = solve_adaptive(a, b, y0, 0.1, eps)
    print(f"Кількість точок зі зміненим кроком: {len(x_hist_rk)}")

    # --- ПОБУДОВА ГРАФІКІВ ---
    plt.figure(figsize=(15, 10))

    # 1
    plt.subplot(2, 2, 1)
    x_exact = [a + i*(b-a)/200 for i in range(201)]
    y_exact = [exact_solution(xi) for xi in x_exact]
    plt.plot(x_exact, y_exact, 'k-', linewidth=2, label='Точний розв_язок')
    plt.plot(x_rk, y_rk, 'b--', label='RK4 (h=0.01)')
    plt.plot(x_ad, y_ad, 'r:', label='Адамс (h=0.01)')
    plt.title('Порівняння чисельних розв_язків')
    plt.legend(); plt.grid(True, alpha=0.3)

    # 2
    plt.subplot(2, 2, 2)
    plt.semilogy(x_rk, err_ex_rk, 'b-', label='Похибка RK4')
    plt.semilogy(x_ad, err_ex_ad, 'r-', label='Похибка Адамса')
    plt.title('Локальна похибка (точно)')
    plt.legend(); plt.grid(True, which="both", alpha=0.3)

    # 3
    plt.subplot(2, 2, 3)
    plt.semilogy(x_rk, err_est_rk, 'b-', label='Оцінка Рунге для RK4')
    plt.semilogy(x_ad, err_est_ad, 'r-', label='Оцінка (5/6|C-P|) для Адамса')
    plt.title('Оцінка локальної похибки')
    plt.legend(); plt.grid(True, which="both", alpha=0.3)

    # 4
    plt.subplot(2, 2, 4)
    plt.step(x_hist_rk, h_hist_rk, 'g-', where='post')
    plt.title('Залежність кроку h(x) для RK4')
    plt.xlabel('x'); plt.ylabel('h')
    plt.ylim(0, 0.12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('ode_plots.png', dpi=300)
    print("\n→ Графіки збережено у файл ode_plots.png")
    plt.show()

if __name__ == "__main__":
    main()