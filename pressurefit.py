import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


# --- 1. MODEL DEFINITIONS ---

def linear_model(t, m, c):
    """
    Defines a linear model for the pressure growth phase.
    p(t) = m*t + c
    Here, m = F0 / V
    """
    return m * t + c


def exponential_decay_model(t, pi, tau, p0):
    """
    Defines an exponential decay model for the pressure decrease phase.
    p(t) = (pi - p0) * np.exp(-t / tau) + p0
    """
    return (pi - p0) * np.exp(-t / tau) + p0


# --- 2. ANALYSIS FUNCTIONS ---

def analyze_pressure_growth(filepath, chamber_volume):
    """
    Analyzes the pressure growth data, using pressure errors for a weighted fit.
    """
    print("\n--- Analyzing Pressure Growth Phase ---")


    data = pd.read_csv(filepath, comment='#', header=0, delim_whitespace=True)
    t = data['time']
    p = data['pressure']
    p_err = data['pressure_error']


    params, covariance = curve_fit(linear_model, t, p, sigma=p_err, absolute_sigma=True)
    m_fit, c_fit = params


    errors = np.sqrt(np.diag(covariance))
    m_err, c_err = errors


    f0_growth = m_fit * chamber_volume
    f0_growth_err = m_err * chamber_volume

    print(f"Linear Fit: p(t) = ({m_fit:.4e} ± {m_err:.2e}) * t + ({c_fit:.4e} ± {c_err:.2e})")
    print(f"Angular Coefficient (Slope): {m_fit:.4e} ± {m_err:.2e} (Pa/s)")
    print(f"Calculated Flux (F0) from growth phase: {f0_growth:.4e} ± {f0_growth_err:.2e} Pa·L/s")


    plt.figure(figsize=(10, 6))
    plt.errorbar(t, p, yerr=p_err, fmt='o', color='blue', ecolor='lightblue', capsize=3, label='Experimental Data')
    plt.plot(t, linear_model(t, m_fit, c_fit),
             label=f'Weighted Linear Fit\nSlope = {m_fit:.2e} ± {m_err:.1e} Pa/s',
             color='red', linestyle='--')
    plt.title('Pressure Growth Phase Analysis')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (Pa)')
    plt.grid(True)
    plt.legend()
    plt.show()

    return f0_growth, f0_growth_err


def analyze_pressure_decrease(filepath, chamber_volume, s_nominal):
    """
    Analyzes pressure decrease, using pressure errors for a weighted fit.
    """
    print("\n--- Analyzing Pressure Decrease Phase ---")


    data = pd.read_csv(filepath, comment='#', header=0, delim_whitespace=True)
    t = data['time']
    p = data['pressure']
    p_err = data['pressure_error']

    pi_initial = p.iloc[0]
    fit_func = lambda t_data, tau, p0: exponential_decay_model(t_data, pi_initial, tau, p0)

    p0_guess = p.min()
    tau_guess = t.max() / 2.0
    initial_guesses = [tau_guess, p0_guess]


    params, covariance = curve_fit(fit_func, t, p, p0=initial_guesses, sigma=p_err, absolute_sigma=True)
    tau_fit, p0_fit = params

    errors = np.sqrt(np.diag(covariance))
    tau_err, p0_err = errors

    print(f"Exponential Fit Results:")
    print(f"Fit Constant τ (tau): {tau_fit:.4f} ± {tau_err:.4f} s")
    print(f"Fit Constant p0 (limit pressure): {p0_fit:.4e} ± {p0_err:.2e} Pa")


    s_effective = chamber_volume / tau_fit
    s_effective_err = s_effective * (tau_err / tau_fit)

    f0_decrease = p0_fit * s_effective
    f0_decrease_err = f0_decrease * np.sqrt((p0_err / p0_fit) ** 2 + (s_effective_err / s_effective) ** 2)

    if s_nominal > s_effective:
        conductivity = (s_effective * s_nominal) / (s_nominal - s_effective)
        print(f"Conductivity of the connection (C): {conductivity:.4f} l/s")
    else:
        conductivity = float('inf')
        print("Effective pumping speed is >= nominal speed. Cannot calculate finite conductivity.")

    print(f"\nDerived Quantities:")
    print(f"Effective Pumping Velocity (S): {s_effective:.4f} ± {s_effective_err:.4f} l/s")
    print(f"Calculated Flux (F0) from decrease phase: {f0_decrease:.4e} ± {f0_decrease_err:.2e} Pa·L/s")

    # Plotting with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(t, p, yerr=p_err, fmt='x', color='green', ecolor='lightgreen', capsize=3, label='Experimental Data')
    plt.plot(t, fit_func(t, tau_fit, p0_fit),
             label=f'Weighted Exponential Fit\nτ = {tau_fit:.2f} ± {tau_err:.2f} s\np0 = {p0_fit:.2e} ± {p0_err:.1e} Pa',
             color='purple', linestyle='--')
    plt.title('Pressure Decrease Phase Analysis')
    plt.xlabel('Time (s)')
    plt.ylabel('Pressure (Pa)')
    #plt.yscale('log')
    #plt.xscale('log')
    plt.grid(True, which="both", ls="-")
    plt.legend()
    plt.show()

    return s_effective, s_effective_err, f0_decrease, f0_decrease_err


# --- 3. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    print("--- Vacuum System Analysis Script ---")

    growth_file_default = 'pressure_growth_data.txt'
    decrease_file_default = 'pressure_decrease_data.txt'


    if not os.path.exists(growth_file_default):
        print(f"Creating sample data file: {growth_file_default}")
        time_g = np.linspace(0, 100, 50)
        pressure_g = 1e-4 * time_g + 5e-3 + np.random.normal(0, 2e-4, 50)
        pressure_g_err = np.full_like(pressure_g, 2.5e-4)  # Constant error for sample
        np.savetxt(growth_file_default, np.c_[time_g, pressure_g, pressure_g_err], fmt='%.6e',
                   header='time pressure pressure_error', comments='')

    if not os.path.exists(decrease_file_default):
        print(f"Creating sample data file: {decrease_file_default}")
        time_d = np.linspace(0, 300, 100)
        pi_d, p0_d, tau_d = 0.1, 5e-5, 50
        pressure_d = (pi_d - p0_d) * np.exp(-time_d / tau_d) + p0_d + np.random.normal(0, 1e-6, 100)
        pressure_d_err = np.full_like(pressure_d, 2e-6)  # Constant error for sample
        np.savetxt(decrease_file_default, np.c_[time_d, pressure_d, pressure_d_err], fmt='%.6e',
                   header='time pressure pressure_error', comments='')

    growth_filepath = input(
        f"Enter the path for the pressure growth data file [{growth_file_default}]: ") or growth_file_default
    decrease_filepath = input(
        f"Enter the path for the pressure decrease data file [{decrease_file_default}]: ") or decrease_file_default

    chamber_volume_V = float(input("Enter the chamber volume (V) in liters [100.0]: ") or 100.0)
    pump_speed_S_nominal = float(input("Enter the nominal pump speed in l/s [33.0]: ") or 33.0)

    try:
        f0_growth, f0_growth_err = analyze_pressure_growth(growth_filepath, chamber_volume_V)
        s_eff, s_eff_err, f0_decrease, f0_decrease_err = analyze_pressure_decrease(decrease_filepath, chamber_volume_V,
                                                                                   pump_speed_S_nominal)

        f_comp=np.abs(f0_growth-f0_decrease)/(np.sqrt(f0_growth_err**2+f0_decrease_err**2))
        print("\n--- FINAL RESULTS & COMPARISON ---")
        print(f"Flux from Growth Phase (F0): {f0_growth:.4e} ± {f0_growth_err:.2e} Pa·L/s")
        print(f"Flux from Decrease Phase (F0): {f0_decrease:.4e} ± {f0_decrease_err:.2e} Pa·L/s")
        print(f"Compatiblity coefficient of F0 values: {f_comp}")

        print(f"\nNominal Pumping Speed: {pump_speed_S_nominal} l/s")
        print(f"Effective Pumping Speed (S_eff): {s_eff:.4f} ± {s_eff_err:.4f} l/s")

        efficiency = (s_eff / pump_speed_S_nominal) * 100
        print(f"Pumping Efficiency: {efficiency:.2f}%")

    except FileNotFoundError as e:
        print(f"\nError: File not found. Please check the file path: {e.filename}")
    except KeyError as e:
        print(
            f"\nError: A required column is missing from your data file: {e}. Please ensure the file has 'time', 'pressure', and 'pressure_error' columns.")
    except Exception as e:
        print(f"\nAn error occurred during analysis: {e}")