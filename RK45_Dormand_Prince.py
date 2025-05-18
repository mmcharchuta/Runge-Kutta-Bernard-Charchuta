import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

# Example ODE: dy/dt = -2y, y(0) = 1
def dydt(t, y):
    return -2 * y

t_span = (0, 5)
y0 = [1]
t_eval = np.linspace(t_span[0], t_span[1], 100)

sol = solve_ivp(dydt, t_span, y0, method='RK45', t_eval=t_eval)

plt.plot(sol.t, sol.y[0], label='RK45 Solution')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Solution of dy/dt = -2y using RK45')
plt.legend()
plt.grid(True)
plt.show()