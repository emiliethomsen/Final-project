import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit
from scipy.stats import t

# ==========================================================================
#                       PARAMETERS
# ==========================================================================

# Define parameters
kws = 1440    # uptake rate constant from water (d^-1)
ksw = 0.553    # elimination rate constant to water (d^-1)
Cw = 1.0     # concentration in water (mmol/L)

bw = 0.26     # killing rate constant (L mmol^-1 d^-1)
hb = 4.5e-3   # background hazard rate (d^-1)
zw = 7.94     # median no effect concentration (mmol/L)

# ==========================================================================
#                       INITITAL CONDITIONS
# ==========================================================================

# Initial conditions
Cs0 = 0.0    # initial concentration in structural layer
S0 = 1.0     # initial survival probability

# Time span (days)
time = np.linspace(0, 2, 1000)
time2 = np.linspace(0, 10, 1000)

# Timepoints of interest (in days)
timepoints_minutes = [1, 5, 10, 30, 60, 300, 960, 1440]  # minutes
timepoints_days = [tt / 1440 for tt in timepoints_minutes]

# ==========================================================================
#                       MODEL SETUP
# ==========================================================================

def hazard(bw, Cs, zw, hb):
    return bw * np.maximum(0, Cs - zw) + hb

def model(y, time, kws, ksw, bw, hb, zw, Cw):
    Cs, S = y
    dCs_dt = kws * Cw - ksw * Cs
    hz = hazard(bw, Cs, zw, hb)
    dS_dt = -hz * S
    return [dCs_dt, dS_dt]

args = (kws, ksw, bw, hb,zw,Cw)
y0 = [Cs0, S0]

# Solve the system
sol = odeint(model, y0, time, args=args)


# Extract solutions
Cs = sol[:, 0]
S = sol[:, 1]

#Calculate hazard over time
hz_time = np.array([hazard(bw, cs, zw, hb) for cs in Cs])


# ==========================================================================
#                       PLOTTING
# ==========================================================================

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Concentrations
axs[0].plot(time, Cs, label='Structural Layer Concentration ($C_i$)')
axs[0].set_ylabel('Concentration (mmol L$^{-1}$)')
axs[0].set_xlabel('Time (days)')
axs[0].legend()
axs[0].grid()

# Hazard
axs[1].plot(time, hz_time, label='Hazard ($h_z$)', color='red')
axs[1].set_ylabel('Hazard rate (d$^{-1}$)')
axs[1].set_xlabel('Time (days)')
axs[1].legend()
axs[1].grid()

# Survival
axs[2].plot(time, S, label='Survival Probability (S)', color='green')
axs[2].set_ylabel('Survival Probability')
axs[2].set_xlabel('Time (days)')
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()

# ==========================================================================
#                       STEADY STATE
# ==========================================================================

# Re-solve the system for a longer time to reach steady state
sol2 = odeint(model, y0, time2, args=args)

# Extract new solutions
Cs2 = sol2[:, 0]
S2 = sol2[:, 1]
hz_time2 = np.array([hazard(bw, cs, zw, hb) for cs in Cs2])

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Concentrations
axs[0].plot(time2, Cs2, label='Structural Layer Concentration ($C_s$)')
axs[0].set_ylabel('Concentration (mmol L$^{-1}$)')
axs[0].set_xlabel('Time (days)')
axs[0].legend()
axs[0].grid()

# Hazard
axs[1].plot(time2, hz_time2, label='Hazard ($h_z$)', color='red')
axs[1].set_ylabel('Hazard rate (d$^{-1}$)')
axs[1].set_xlabel('Time (days)')
axs[1].legend()
axs[1].grid()

# Survival
axs[2].plot(time2, S2, label='Survival Probability (S)', color='green')
axs[2].set_ylabel('Survival Probability')
axs[2].set_xlabel('Time (days)')
axs[2].legend()
axs[2].grid()

plt.tight_layout()
plt.show()

Ceq = (kws*Cw)/ksw
hzeq = bw * max(0,Ceq - zw) + hb
Seq = 1/hzeq

# Define a threshold for when we consider the system to have reached Ceq.
# For example, when Cs is within 1% of Ceq.
threshold = 0.01 * Ceq  # 1% of Ceq

# Find the time when Cs first enters the steady-state range (within the threshold of Ceq)
steady_state_time = time2[np.where(np.abs(Cs2 - Ceq) < threshold)[0][0]]

print(f'Time = {steady_state_time:.2f} days')
print(f'Cs = {Ceq:.2f}')
print(f'hz = {hzeq:.2f}')
print(f'S = {Seq}')
# ==========================================================================
#                       GET SURVIVAL DATA FOR KWS
# ==========================================================================

kws_values = [1440, 2880, 5760, 11520]
colors = plt.cm.viridis(np.linspace(0, 1, len(kws_values)))

fig, axs = plt.subplots(3, 1, figsize=(10, 12))
survival_table = {}

for i, kws in enumerate(kws_values):
    args = (kws, ksw, bw, hb, zw, Cw)
    y0 = [Cs0, S0]
    sol = odeint(model, y0, time, args=args)

    Cs = sol[:, 0]
    S = sol[:, 1]
    hz_time = np.array([hazard(bw, cs, zw, hb) for cs in Cs])

    label = f"kws = {kws}"
    axs[0].plot(time, Cs,label=label, color=colors[i])
    axs[1].plot(time, hz_time, label=label, color=colors[i])
    axs[2].plot(time, S, label=label, color=colors[i])

    survival_table[label] = [S[np.argmin(np.abs(time - tt))] for tt in timepoints_days]

for ax, title, ylabel in zip(
    axs,
    ["Internal Concentration", "Hazard Rate", "Survival Probability"],
    ['$C_i$ (mmol/L)', 'Hazard Rate (d$^{-1}$)', 'Survival Probability']):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (days)')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()

# ==========================================================================
#                       ESTIMATE KWS
# ==========================================================================

def solve_kws(timepoints, kws, ksw, bw, hb, zw, Cw):
    y0 = [Cs0, S0]
    sol = odeint(model, y0, timepoints, args=(kws, ksw, bw, hb, zw, Cw))
    return sol[:, 1]

true_kws_values = kws_values
observed_survival_data = [survival_table[f"kws = {kws}"] for kws in kws_values]
estimated_kws_values = []
cov_matrices = []

for kws_true, observed_data in zip(true_kws_values, observed_survival_data):
    popt, pcov = curve_fit(
        lambda tt, kws: solve_kws(tt, kws, ksw, bw, hb, zw, Cw),
        timepoints_days, observed_data, p0=[kws_true], bounds=(0, np.inf)
    )
    estimated_kws_values.append(popt[0])
    cov_matrices.append(pcov)
    
    
# ==========================================================================
#                       STATISTICAL TESTING KWS
# ==========================================================================

n = len(timepoints_days)
dof = n - 1

print(f"{'True kws':<10} {'Est. kws':<10} {'SE':<10} {'t-stat':<10} {'p-value':<10} {'Significant?'}")
print("-" * 70)

for true_kws, est_kws, pcov in zip(true_kws_values, estimated_kws_values, cov_matrices):
    se = np.sqrt(np.diag(pcov))[0]
    t_stat = (est_kws - true_kws) / se
    p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=dof))
    significant = "Yes" if p_val < 0.05 else "No"
    print(f"{true_kws:<10.3f} {est_kws:<10.3f} {se:<10.4f} {t_stat:<10.3f} {p_val:<10.7f} {significant}")
    
# ==========================================================================
#                       GET SURVIVAL DATA FOR KSW
# ==========================================================================

kws = 1440
ksw_values = [0.553, 1.106, 2.212, 4.424]
colors = plt.cm.viridis(np.linspace(0, 1, len(ksw_values)))

fig, axs = plt.subplots(3, 1, figsize=(10, 12))
survival_table = {}

for i, ksw in enumerate(ksw_values):
    args = (kws, ksw, bw, hb, zw, Cw)
    y0 = [Cs0, S0]
    sol = odeint(model, y0, time, args=args)

    Cs = sol[:, 0]
    S = sol[:, 1]
    hz_time = np.array([hazard(bw, cs, zw, hb) for cs in Cs])

    label = f"ksw = {ksw}"
    axs[0].plot(time, Cs, label=label, color=colors[i])
    axs[1].plot(time, hz_time, label=label, color=colors[i])
    axs[2].plot(time, S, label=label, color=colors[i])

    survival_table[label] = [S[np.argmin(np.abs(time - tt))] for tt in timepoints_days]

for ax, title, ylabel in zip(
    axs,
    ["Internal Concentration", "Hazard Rate", "Survival Probability"],
    ['$C_i$ (mmol/L)', 'Hazard Rate (d$^{-1}$)', 'Survival Probability']):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (days)')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()

# ==========================================================================
#                       ESTIMATE KSW
# ==========================================================================

def solve_ksw(timepoints, kws, ksw, bw, hb, zw, Cw):
    y0 = [Cs0, S0]
    sol = odeint(model, y0, timepoints, args=(kws, ksw, bw, hb, zw, Cw))
    return sol[:, 1]

true_ksw_values = ksw_values
observed_survival_data = [survival_table[f"ksw = {ksw}"] for ksw in ksw_values]
estimated_ksw_values = []
cov_matrices = []

for ksw_true, observed_data in zip(true_ksw_values, observed_survival_data):
    popt, pcov = curve_fit(
        lambda tt, ksw: solve_ksw(tt, kws, ksw, bw, hb, zw, Cw),
        timepoints_days, observed_data, p0=[ksw_true], bounds=(0, np.inf)
    )
    estimated_ksw_values.append(popt[0])
    cov_matrices.append(pcov)

    
# ==========================================================================
#                       STATISTICAL TESTING KSW
# ==========================================================================

n = len(timepoints_days)
dof = n - 1

print(f"{'True ksw':<10} {'Est. ksw':<10} {'SE':<10} {'t-stat':<10} {'p-value':<10} {'Significant?'}")
print("-" * 70)

for true_ksw, est_ksw, pcov in zip(true_ksw_values, estimated_ksw_values, cov_matrices):
    se = np.sqrt(np.diag(pcov))[0]
    t_stat = (est_ksw - true_ksw) / se
    p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=dof))
    significant = "Yes" if p_val < 0.05 else "No"
    print(f"{true_ksw:<10.3f} {est_ksw:<10.6f} {se:<10.4f} {t_stat:<10.3f} {p_val:<10.7f} {significant}")
