import numpy as np
import matplotlib.pyplot as plt


def f(t, state):
    x, y = state
    dx = -x + x * y
    dy = 2 * y + x**2
    return np.array([dx, dy])


def rk4_step(t, state, dt):
    k1 = f(t, state)
    k2 = f(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = f(t + dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate(state0, t_span=(0.0, 2.0), dt=0.01):
    t0, t1 = t_span
    steps = int(np.ceil((t1 - t0) / dt))
    t = t0
    state = np.array(state0, dtype=float)
    traj = np.empty((steps + 1, 2))
    traj[0] = state
    for i in range(steps):
        state = rk4_step(t, state, dt)
        t += dt
        traj[i + 1] = state
    return traj


def main():
    initial_conditions = [
        (0.1, 0.1),
        (-0.1, 0.1),
        (0.1, -0.1),
        (-0.1, -0.1),
        (0.2, 0.0),
        (0.0, 0.2),
        (0.15, -0.05),
        (-0.15, 0.05),
    ]

    plt.figure(figsize=(6, 5))

    for state0 in initial_conditions:
        traj = integrate(state0)
        plt.plot(traj[:, 0], traj[:, 1], linewidth=1.5)

    x = np.linspace(-0.3, 0.3, 200)
    y = -0.25 * x**2
    plt.plot(x, y, 'k--', label=r'$y=-\frac{1}{4}x^2$')

    plt.title('Phase Portrait Near the Origin')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-0.3, 0.3)
    plt.ylim(-0.3, 0.3)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase_plot.png', dpi=200)
    plt.show()


if __name__ == '__main__':
    main()
