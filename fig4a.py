import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# System parameters from Cao et al. (2006) [1]
f0 = 0.8
xi = 0.01 * np.sqrt(2)
omega = 0.75 * np.sqrt(2)

def system(t, X, alpha):
    x, y = X
    dxdt = y
    dydt = -2*xi*y - x*(1 - 1/np.sqrt(x**2 + alpha**2)) + f0*np.cos(omega*t)
    return [dxdt, dydt]

def period1_bifurcation():
    # Alpha range (decreasing from 1 to 0)
    alpha_values = np.linspace(1, 0, 100)
    
    # Initial conditions for period-1 [1]
    x0, y0 = 0.8, 0  
    
    # Store results
    bifurcation_data = []

    current_x, current_y = x0, y0
    for alpha in alpha_values:
        # Compute stroboscopic points
        t_span = [0, 200 * 2 * np.pi / omega]
        sol = solve_ivp(lambda t,X: system(t,X,alpha), t_span, 
                       [current_x, current_y], method='RK45', 
                       dense_output=True, rtol=1e-8, atol=1e-10)
        
        # Get stroboscopic samples (after transient)
        strob_times = np.arange(150, 200) * 2 * np.pi / omega
        x_points = sol.sol(strob_times)[0]
        
        bifurcation_data.extend([(alpha, x) for x in x_points])
        
        # Update initial conditions using final state
        current_x, current_y = sol.y[0,-1], sol.y[1,-1]

    # Plot results
    alphas, xs = zip(*bifurcation_data)
    plt.figure(figsize=(10,6))
    plt.plot(alphas, xs, '.k', markersize=1, alpha=0.5)
    plt.xlabel(r'$\alpha$', fontsize=14)
    plt.ylabel('x', fontsize=14)
    plt.title('Period-1 Bifurcation Diagram', fontsize=16)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    period1_bifurcation()
