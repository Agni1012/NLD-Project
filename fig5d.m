% Parameters
zeta = 0.01 * sqrt(2);        % damping
f0 = 0.8;                     % forcing amplitude
Omega = 0.75 * sqrt(2);       % forcing frequency
eta = 0.5;                    % smoothness parameter

% Driving period
T_drive = 2 * pi / Omega;

% Time configuration
T_total = 3000 * T_drive;         % 3000 cycles = 3000 points on plot
N_steps_per_cycle = 1000;
N_steps_total = T_total / T_drive * N_steps_per_cycle;
tspan = linspace(0, T_total, N_steps_total);  % dense time vector

% Initial condition
y0 = [1.0; 0];

% Define oscillator
oscillator = @(t, y) [
    y(2);
    -2*zeta*y(2) - y(1)*(1 - 1/sqrt(y(1)^2 + eta^2)) + f0 * cos(Omega * t)
];

% Solve ODE
opts = odeset('RelTol',1e-9,'AbsTol',1e-11);
[t, Y] = ode45(oscillator, tspan, y0, opts);

% Stroboscopic sampling (discard transients)
transient_cut = 100 * T_drive;
strobe_times = transient_cut:T_drive:T_total;
x_strobe = interp1(t, Y(:,1), strobe_times);
dx_strobe = interp1(t, Y(:,2), strobe_times);

% Plot
figure;
plot(x_strobe, dx_strobe, 'b.', 'MarkerSize', 1.5)
xlabel('x')
ylabel('dx/dt')
title('Phase-space plot (Poincaré section, \eta = 0.5)')
grid on
axis equal
