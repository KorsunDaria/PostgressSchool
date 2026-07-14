//python3 Day2_task4.py --lam 9 --mu 4 --k 3 --n 200000

import argparse
import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


# ---------------------------------------------------------------------------
# 1. Дискретно-событийная симуляция M/M/k
# ---------------------------------------------------------------------------
def simulate_mmk(lam: float, mu: float, k: int, n_customers: int, seed: int = 42):
 
    rng = np.random.default_rng(seed)

    interarrival = rng.exponential(1 / lam, n_customers)
    service = rng.exponential(1 / mu, n_customers)
    arrival = np.cumsum(interarrival)

    start = np.zeros(n_customers)
    depart = np.zeros(n_customers)

    servers = [0.0] * k          # все k серверов изначально свободны
    heapq.heapify(servers)

    for i in range(n_customers):
        free_at = heapq.heappop(servers)          # самый скоро свободный сервер
        s = max(arrival[i], free_at)               # заявка не может начаться раньше прихода
        d = s + service[i]
        start[i] = s
        depart[i] = d
        heapq.heappush(servers, d)                 # сервер теперь занят до d

    wait_q = start - arrival
    time_sys = depart - arrival
    return arrival, start, depart, wait_q, time_sys


def build_number_in_system(arrival, depart):
    events = np.concatenate([arrival, depart])
    deltas = np.concatenate([np.ones_like(arrival), -np.ones_like(depart)])
    order = np.argsort(events, kind="mergesort")
    events, deltas = events[order], deltas[order]

    n = np.cumsum(deltas)
    t = events
    durations = np.diff(t, append=t[-1])
    L_time_avg = np.sum(n * durations) / t[-1]

    return t, n, L_time_avg


# ---------------------------------------------------------------------------
# 2. Теоретические формулы M/M/k (Эрланг C)
# ---------------------------------------------------------------------------
def theoretical_mmk(lam, mu, k):
    a = lam / mu                      # предлагаемая нагрузка
    rho = a / k

    # сумма для P0
    terms = [a ** n / math.factorial(n) for n in range(k)]
    last_term = (a ** k) / (math.factorial(k) * (1 - rho))
    P0 = 1 / (sum(terms) + last_term)

    erlang_c = last_term * P0          # вероятность ожидания (формула Эрланга C)

    Lq = erlang_c * rho / (1 - rho)
    Wq = Lq / lam
    W = Wq + 1 / mu
    L = lam * W

    return dict(a=a, rho=rho, P0=P0, erlang_c=erlang_c, L=L, Lq=Lq, W=W, Wq=Wq)


# ---------------------------------------------------------------------------
# 3. Красивый вывод в консоль
# ---------------------------------------------------------------------------
def print_report(lam, mu, k, sim, theory):
    console.print(Panel.fit(
        f"[bold]M/M/{k}[/bold]   λ = {lam}   μ = {mu}   "
        f"a = λ/μ = {lam/mu:.3f}   ρ = a/k = {theory['rho']:.3f}   "
        f"P(ожидание) = {theory['erlang_c']:.3f}",
        style="cyan"))

    table = Table(title="Сравнение: симуляция vs теория (Эрланг C)", show_lines=True)
    table.add_column("Показатель", style="bold")
    table.add_column("Симуляция", justify="right")
    table.add_column("Теория", justify="right")
    table.add_column("Отклонение", justify="right")

    rows = [
        ("L  (среднее число в системе)", sim["L"], theory["L"]),
        ("Lq (среднее число в очереди)", sim["Lq"], theory["Lq"]),
        ("W  (среднее время в системе)", sim["W"], theory["W"]),
        ("Wq (среднее время в очереди)", sim["Wq"], theory["Wq"]),
    ]
    for name, s, t in rows:
        dev = abs(s - t) / t * 100 if t > 0 else float("nan")
        table.add_row(name, f"{s:.4f}", f"{t:.4f}", f"{dev:.2f}%")

    console.print(table)

    littles = Table(title="Проверка формулы Литтла (L = λ·W)", show_lines=True)
    littles.add_column("Соотношение")
    littles.add_column("Левая часть", justify="right")
    littles.add_column("Правая часть", justify="right")
    littles.add_column("Расхождение", justify="right")

    L_from_W = lam * sim["W"]
    Lq_from_Wq = lam * sim["Wq"]

    littles.add_row("L  =  λ·W", f"{sim['L']:.4f}", f"{L_from_W:.4f}",
                     f"{abs(sim['L']-L_from_W):.5f}")
    littles.add_row("Lq =  λ·Wq", f"{sim['Lq']:.4f}", f"{Lq_from_Wq:.4f}",
                     f"{abs(sim['Lq']-Lq_from_Wq):.5f}")

    console.print(littles)


# ---------------------------------------------------------------------------
# 4. Графики
# ---------------------------------------------------------------------------
def make_plots(lam, mu, k, arrival, depart, wait_q, time_sys, t_n, n_vals,
               n_customers, outfile):

    theory = theoretical_mmk(lam, mu, k)
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"Модель M/M/{k}:  λ={lam}, μ={mu}, ρ={theory['rho']:.2f}   "
                 f"(N={n_customers} заявок)", fontsize=15, fontweight="bold")

    # --- (1) число заявок в системе во времени ---------------------------
    ax1 = plt.subplot(2, 3, 1)
    show = min(len(t_n), 4000)
    ax1.step(t_n[:show], n_vals[:show], where="post", color="#2563eb", lw=0.8)
    ax1.axhline(theory["L"], color="crimson", ls="--", lw=1.5,
                label=f"теор. L={theory['L']:.2f}")
    ax1.axhline(k, color="gray", ls=":", lw=1, label=f"k={k} серверов")
    ax1.set_title("Число заявок в системе N(t)")
    ax1.set_xlabel("время")
    ax1.set_ylabel("N(t)")
    ax1.legend()

    # --- (2) сходимость оценки L ------------------------------------------
    ax2 = plt.subplot(2, 3, 2)
    durations = np.diff(t_n, append=t_n[-1])
    cum_area = np.cumsum(n_vals * durations)
    running_L = cum_area / t_n
    ax2.plot(t_n, running_L, color="#16a34a", lw=1)
    ax2.axhline(theory["L"], color="crimson", ls="--", lw=1.5,
                label=f"теор. L={theory['L']:.2f}")
    ax2.set_title("Сходимость оценки L(t) к теоретическому значению")
    ax2.set_xlabel("время")
    ax2.set_ylabel("текущая оценка L")
    ax2.legend()

    # --- (3) гистограмма времени в очереди --------------------------------
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(wait_q, bins=50, density=True, color="#f59e0b", alpha=0.75,
              label="симуляция Wq")
    # для M/M/k: с вероятностью (1-C) заявка не ждёт вовсе (масса в 0),
    # а с вероятностью C время ожидания ~ Exp(k*mu - lam)
    C = theory["erlang_c"]
    rate = k * mu - lam
    xs = np.linspace(0.001, wait_q.max() if wait_q.max() > 0 else 1, 300)
    pdf = C * rate * np.exp(-rate * xs)
    ax3.plot(xs, pdf, color="crimson", lw=2, label="теор. плотность (>0 часть)")
    ax3.set_title("Распределение времени ожидания в очереди")
    ax3.set_xlabel("Wq")
    ax3.legend()

    # --- (4) гистограмма времени в системе --------------------------------
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(time_sys, bins=50, density=True, color="#8b5cf6", alpha=0.75,
              label="симуляция W")
    ax4.set_title("Распределение времени пребывания в системе")
    ax4.set_xlabel("W")
    ax4.axvline(theory["W"], color="crimson", lw=2, ls="--",
                label=f"теор. среднее W={theory['W']:.2f}")
    ax4.legend()

    # --- (5) столбчатая диаграмма: симуляция vs теория --------------------
    ax5 = plt.subplot(2, 3, 5)
    sim = dict(
        L=running_L[-1],
        Lq=np.sum(np.maximum(n_vals - k, 0) * durations) / t_n[-1],
        W=time_sys.mean(),
        Wq=wait_q.mean(),
    )
    labels = ["L", "Lq", "W", "Wq"]
    sim_vals = [sim[l] for l in labels]
    th_vals = [theory[l] for l in labels]
    x = np.arange(len(labels))
    width = 0.35
    ax5.bar(x - width / 2, sim_vals, width, label="Симуляция", color="#2563eb")
    ax5.bar(x + width / 2, th_vals, width, label="Теория", color="#f97316")
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.set_title("Симуляция vs теория")
    ax5.legend()
    for i, (sv, tv) in enumerate(zip(sim_vals, th_vals)):
        ax5.text(i - width / 2, sv, f"{sv:.2f}", ha="center", va="bottom", fontsize=8)
        ax5.text(i + width / 2, tv, f"{tv:.2f}", ha="center", va="bottom", fontsize=8)

    # --- (6) проверка формулы Литтла на скользящем окне --------------------
    ax6 = plt.subplot(2, 3, 6)
    window = max(n_customers // 40, 20)
    LambdaW_local, idx = [], []
    for i in range(window, n_customers, window):
        seg_arr = arrival[i - window:i]
        seg_w = time_sys[i - window:i]
        span = seg_arr[-1] - seg_arr[0]
        if span <= 0:
            continue
        lam_local = window / span
        LambdaW_local.append(lam_local * seg_w.mean())
        idx.append(i)
    ax6.plot(idx, LambdaW_local, "o-", color="#0891b2", ms=3,
             label=r"локальная оценка $\lambda \cdot W$")
    ax6.axhline(theory["L"], color="crimson", ls="--", lw=1.5,
                label=f"теор. L={theory['L']:.2f}")
    ax6.set_title("Проверка Литтла по скользящим окнам")
    ax6.set_xlabel("номер заявки")
    ax6.set_ylabel(r"$\lambda \cdot W$")
    ax6.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outfile, dpi=140)
    console.print(f"[green]График сохранён в[/green] {outfile}")
    return sim


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Симуляция M/M/k и проверка формул Эрланга C / Литтла")
    parser.add_argument("--lam", type=float, default=9.0, help="интенсивность входного потока λ")
    parser.add_argument("--mu", type=float, default=4.0, help="интенсивность обслуживания одного сервера μ")
    parser.add_argument("--k", type=int, default=3, help="число серверов")
    parser.add_argument("--n", type=int, default=200_000, help="число заявок в симуляции")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="mmk_result.png")
    args = parser.parse_args()

    if args.lam >= args.k * args.mu:
        raise SystemExit("Ошибка: для стационарности системы должно быть λ < k·μ (ρ < 1)")

    arrival, start, depart, wait_q, time_sys = simulate_mmk(
        args.lam, args.mu, args.k, args.n, args.seed)
    t_n, n_vals, _ = build_number_in_system(arrival, depart)

    theory = theoretical_mmk(args.lam, args.mu, args.k)
    sim = make_plots(args.lam, args.mu, args.k, arrival, depart, wait_q, time_sys,
                      t_n, n_vals, args.n, args.out)

    print_report(args.lam, args.mu, args.k, sim, theory)


if __name__ == "__main__":
    main()