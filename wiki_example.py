import math
import matplotlib.pyplot as plt

g = 9.81  # Przyspieszenie ziemskie
L = 9.81   # Długość wahadła

T_0 = 2 * math.pi * math.sqrt(L / g)

def f(theta, omega, c):
    # Równania ruchu dla wahadła z tłumieniem
    omega_dot = (-c * omega - (g / L) * math.sin(theta))
    return omega_dot

def runge_kutta(theta, omega, h, c):
    k1_theta = h * omega
    k1_omega = h * f(theta, omega, c)

    k2_theta = h * (omega + k1_omega / 2)
    k2_omega = h * f(theta + k1_theta / 2, omega + k1_omega / 2, c)

    k3_theta = h * (omega + k2_omega / 2)
    k3_omega = h * f(theta + k2_theta / 2, omega + k2_omega / 2, c)

    k4_theta = h * (omega + k3_omega)
    k4_omega = h * f(theta + k3_theta, omega + k3_omega, c)

    theta += (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta) / 6
    omega += (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega) / 6

    return theta, omega

def main():
    # DANE:
    theta_0 = 170  # Początkowy kąt wychylenia [w stopniach]

    h = 0.00001          # Krok czasu
    iterations = 4000000  # Liczba iteracji

    c_values = [0, 0.2, 0.5, 1.5]  # Wartości współczynnika tłumienia c
    print(c_values)

    for c in c_values:
        omega = 0
        theta = theta_0 / 180 * math.pi
        time_values = []
        theta_values = []

        print("Tłumienie c =", c)
        time_2 = 0
        for i in range(iterations):
            time = i * h
            time_values.append(time / T_0)
            theta_values.append(theta)

            omega_2 = omega
            theta, omega = runge_kutta(theta, omega, h, c)
            if (omega * omega_2 < 0) or (omega_2 == 0):
                print("T/2/T_0=", (time - time_2) / T_0)
                time_2 = time

        plt.plot(time_values, theta_values)

    # WYKRESY - RYSOWANIE:
    plt.xlabel(r'Czas unormowany $t/T_0$')
    plt.ylabel(r'$\theta(t)$ [stopnie]')
    plt.title('Wykres kąta wychylenia wahadła z tłumieniem od czasu')
    plt.grid(True)
    plt.legend(['c = 0', 'c = 0.2', 'c = 0.5', 'c = 1.5'])

    plt.show()

if __name__ == "__main__":
    main()