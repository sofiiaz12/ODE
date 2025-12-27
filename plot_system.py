import numpy as np
import matplotlib.pyplot as plt


def system_rhs(x, y):
    dx = -x + x * y
    dy = 2 * y + x**2
    return dx, dy


def rk4_step(x, y, dt):
    k1x, k1y = system_rhs(x, y)
    k2x, k2y = system_rhs(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
    k3x, k3y = system_rhs(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
    k4x, k4y = system_rhs(x + dt * k3x, y + dt * k3y)
    x_next = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    y_next = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
    return x_next, y_next


def integrate_trajectory(x0, y0, t_end, dt, max_abs=1e3):
    steps = int(t_end / dt)
    xs = np.empty(steps + 1)
    ys = np.empty(steps + 1)
    xs[0] = x0
    ys[0] = y0
    x = x0
    y = y0
    for i in range(steps):
        x, y = rk4_step(x, y, dt)
        xs[i + 1] = x
        ys[i + 1] = y
        if not np.isfinite(x) or not np.isfinite(y) or abs(x) > max_abs or abs(y) > max_abs:
            return xs[: i + 2], ys[: i + 2]
    return xs, ys


def main():
    initial_conditions = [
        (-3.0, -1.0),
        (-2.0, 0.5),
        (-1.0, -0.5),
        (1.0, -0.5),
        (2.0, 0.5),
        (3.0, 2.0),
    ]
    t_end = 5.0
    dt = 0.01

    fig, ax = plt.subplots(figsize=(8, 6))

    for x0, y0 in initial_conditions:
        xs, ys = integrate_trajectory(x0, y0, t_end, dt)
        ax.plot(xs, ys, label=f"({x0}, {y0})")

    x_vals = np.linspace(-4.0, 4.0, 400)
    y_vals = -0.25 * x_vals**2
    ax.plot(x_vals, y_vals, "k--", linewidth=2, label=r"$y=-\frac{1}{4}x^2$")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Numerical solutions in the phase plane")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 6)
    fig.tight_layout()
    fig.savefig("phase_plot.png", dpi=150)


if __name__ == "__main__":
    main()
