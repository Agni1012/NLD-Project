import numpy as np
import matplotlib.pyplot as plt

# System parameters
delta = 0.05
zeta = 0.01
f0 = 0.8
Omega = 0.75 * np.sqrt(2)
sigma = 0.05  # You can sweep this later to study its effect

# Simulation parameters
T = 500
dt = 0.01
N = int(T / dt)
time = np.linspace(0, T, N)

# Initialize state arrays
x = np.zeros(N)
v = np.zeros(N)

# Initial conditions
x[0] = 1.0
v[0] = 0.0

# Integrate using Euler-Maruyama
for i in range(N - 1):
    dxdt = v[i]
    dvdt = -2*zeta*v[i] - x[i]*(1 - 1/(x[i]**2 + delta**2)) + f0 * np.cos(Omega * time[i])
    v[i+1] = v[i] + dvdt * dt + sigma * np.sqrt(dt) * np.random.randn()
    x[i+1] = x[i] + dxdt * dt

# Detect crossings from x<0 to x>0 or vice versa
signs = np.sign(x)
switches = np.where(np.diff(signs))[0]  # indices where sign changes
switch_times = time[switches]

# Calculate switching rate (switches per unit time)
total_switches = len(switches)
switching_rate = total_switches / T

# Plotting results
plt.figure(figsize=(12, 5))

# Time series with switches
plt.subplot(1, 2, 1)
plt.plot(time, x, label='x(t)')
plt.plot(switch_times, x[switches], 'ro', label='Switches', markersize=3)
plt.title("Time Series with Switching Points")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.legend()

# Histogram of inter-switch times
if len(switch_times) > 1:
    intervals = np.diff(switch_times)
    plt.subplot(1, 2, 2)
    plt.hist(intervals, bins=50, color='skyblue', edgecolor='k')
    plt.title("Histogram of Inter-switch Intervals")
    plt.xlabel("Time between switches")
    plt.ylabel("Count")

plt.tight_layout()
plt.show()

print(f"Total switches: {total_switches}")
print(f"Switching rate: {switching_rate:.4f} switches per unit time")
