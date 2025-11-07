import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os



try:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
    })
    print("Matplotlib style updated to use LaTeX for rendering.")
    print("Ensure a LaTeX distribution (like TeX Live or MiKTeX) is installed.")
except Exception as e:
    print(f"--- Warning: Failed to set LaTeX rendering for plots. ---")
    print(f"Error: {e}")
    print("Falling back to default. Plots will not have TeX formatting.")




def format_sci_pm(value, error):
    """
    Formats a value and error into the LaTeX string:
    (v.v \pm e.v) \cdot 10^{exp}

    The exponent is chosen based on the error's magnitude
    to format the error mantissa as 0.x.
    """
    if error == 0 or not np.isfinite(error):

        if value == 0 or not np.isfinite(value):
            return "(0.0 \pm 0.0) \cdot 10^{0}"

        exponent = int(np.floor(np.log10(np.abs(value))))
        scale = 10 ** exponent
        mantissa_val = value / scale
        return f"({mantissa_val:.1f} \pm 0.0) \cdot 10^{{{exponent}}}"


    exponent = int(np.floor(np.log10(np.abs(error)))) + 1


    scale = 10 ** exponent
    mantissa_val = value / scale
    mantissa_err = error / scale


    if exponent == 0:
        formatted_string = f"({mantissa_val:.1f} \pm {mantissa_err:.1f})"
    else:
        formatted_string = f"({mantissa_val:.1f} \pm {mantissa_err:.1f}) \cdot 10^{{{exponent}}}"
    return formatted_string


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

def driven_oscillation_model(t, m, c, a, omega, phi):
    """
    Defines a model with a linear trend and an oscillation
    with linearly increasing amplitude.
    p(t) = (m*t + c) + (a*t) * sin(omega*t + phi)
    """
    linear_trend = m * t + c
    oscillation = (a * t) * np.sin(omega * t + phi)
    return linear_trend + oscillation


def analyze_pressure_growth(filepath, chamber_volume):
    """
    Analyzes the pressure growth data, using a 1-PARAMETER fit
    where the intercept is FIXED to the first data point.
    Generates a 2-panel plot with the fit and the residuals.
    """
    print("\n--- Analyzing Pressure Growth Phase (Fixed Intercept) ---")

    data = pd.read_csv(filepath, comment='#', header=0, delim_whitespace=True, decimal=',')
    t = data['time']
    p = data['pressure']
    p_err = data['pressure_error']


    p_initial = p.iloc[0]
    t_initial = t.iloc[0]
    p_initial_err = p_err.iloc[0]


    if t_initial != 0:
        print(f"Note: Initial time is t={t_initial} s (not 0).")


    fixed_c_model = lambda t, m: m * (t - t_initial) + p_initial


    params, covariance = curve_fit(fixed_c_model, t, p, sigma=p_err, absolute_sigma=True)


    m_fit = params[0]
    m_err = np.sqrt(np.diag(covariance))[0]


    c_fit = p_initial
    c_err = p_initial_err

    m_str = format_sci_pm(m_fit, m_err)
    c_str = format_sci_pm(c_fit, c_err)
    fit_label = (rf'Linear Fit' +
                 f'\n$F_{{0,g}}/V = {m_str}$ mbar/s' +
                 f'\n$p_{{i,g}} = {c_str}$ mbar')
    # --- END MODIFICATION ---

    f0_growth = m_fit * chamber_volume
    f0_growth_err = m_err * chamber_volume

    # Updated print statement to reflect the new model
    print(f"Linear Fit: p(t) = ({m_fit:.4e} ± {m_err:.2e}) * (t - {t_initial:.1f}) + ({c_fit:.4e} ± {c_err:.2e})")
    print(f"Angular Coefficient (Slope): {m_fit:.4e} ± {m_err:.2e} (mbar/s)")
    print(f"Calculated Flux (F0) from growth phase: {f0_growth:.4e} ± {f0_growth_err:.2e} mbar·L/s")

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8),
                            gridspec_kw={'height_ratios': [3, 1]})


    axs[0].errorbar(t, p, yerr=p_err, fmt='o', color='blue', ecolor='cornflowerblue', markersize=3,capsize=3, label='Experimental Data')
    axs[0].plot(t, linear_model(t, m_fit, c_fit),
                label=fit_label,
                color='red', linestyle='--')
    axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0, -4))
    axs[0].set_title('Pressure Growth Phase Analysis',fontsize='21')
    axs[0].set_ylabel(r'Pressure (mbar)',fontsize='17')
    axs[0].grid(True)
    axs[0].legend(fontsize='17')


    residuals = p - linear_model(t, m_fit, c_fit)
    axs[1].errorbar(t, residuals, yerr=p_err, fmt='o', color='blue',markersize=3, ecolor='cornflowerblue', capsize=3)
    axs[1].axhline(0, color='red', linestyle='--', linewidth=0.8)  # Add a zero line
    axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, -5))
    axs[1].set_xlabel(r'Time (s)',fontsize='21')
    axs[1].set_ylabel(r'Residuals (mbar)',fontsize='21')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('Pgrowth.pdf')
    plt.show()


    return f0_growth, f0_growth_err

def analyze_pressure_decrease(filepath, chamber_volume, s_nominal):
    """
    Analyzes pressure decrease, using pressure errors for a weighted fit.
    Generates a 2-panel plot with the fit and the residuals.
    """
    print("\n--- Analyzing Pressure Decrease Phase ---")

    data = pd.read_csv(filepath, comment='#', header=0, delim_whitespace=True, decimal=',')
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

    tau_str=format_sci_pm(tau_fit, tau_err)
    p0_str=format_sci_pm(p0_fit, p0_err)

    fit_label = (rf'Exponential Fit' +
                 f'\n$\\tau = {tau_str}$ s' +
                 f'\n$p_0 = {p0_str}$ mbar')

    print(f"Exponential Fit Results:")
    print(f"Fit Constant τ (tau): {tau_fit:.4f} ± {tau_err:.4f} s")
    print(f"Fit Constant p0 (limit pressure): {p0_fit:.4e} ± {p0_err:.2e} mbar")

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
    print(f"Calculated Flux (F0) from decrease phase: {f0_decrease:.4e} ± {f0_decrease_err:.2e} mbar·L/s")


    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 8),
                            gridspec_kw={'height_ratios': [3, 1]})

    # --- Top Panel: Main Fit ---
    axs[0].errorbar(t, p, yerr=p_err, fmt='o', color='blue', ecolor='cornflowerblue',markersize=3, capsize=3, label='Experimental Data')
    axs[0].plot(t, fit_func(t, tau_fit, p0_fit),
                label=fit_label,
                color='red', linestyle='--')
    axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0, -4))
    axs[0].set_title('Pressure Decrease Phase Analysis',fontsize='21')
    axs[0].set_ylabel(r'Pressure (mbar)',fontsize='17')
    axs[0].grid(True, which="both", ls="-")
    axs[0].legend(fontsize='19')



    residuals = p - fit_func(t, tau_fit, p0_fit)
    axs[1].errorbar(t, residuals, yerr=p_err, fmt='o',markersize=3, color='blue', ecolor='cornflowerblue', capsize=3)
    axs[1].axhline(0, color='red', linestyle='--', linewidth=0.8)  # Add a zero line
    axs[1].set_xlabel(r'Time (s)',fontsize='21')
    axs[1].set_ylabel(r'Residuals (mbar)',fontsize='21')
    axs[1].grid(True, which="both", ls="-")

    plt.tight_layout()
    plt.savefig('Pdecrease.pdf')
    plt.show()


    return s_effective, s_effective_err, f0_decrease, f0_decrease_err




if __name__ == "__main__":
    print("--- Vacuum System Analysis Script ---")

    growth_file_default = 'pressure_growth_data.txt'
    decrease_file_default = 'pressure_decrease_data.txt'

    if not os.path.exists(growth_file_default):
        print(f"Creating sample data file: {growth_file_default}")
        time_g = np.linspace(0, 100, 50)
        # Sample data now assumes mbar units (oscillation removed)
        pressure_g = 1e-6 * time_g + 5e-5 + np.random.normal(0, 2e-6, 50)
        pressure_g_err = np.full_like(pressure_g, 2.5e-6)
        np.savetxt(growth_file_default, np.c_[time_g, pressure_g, pressure_g_err], fmt='%.6e',
                   header='time pressure pressure_error', comments='')

    if not os.path.exists(decrease_file_default):
        print(f"Creating sample data file: {decrease_file_default}")
        time_d = np.linspace(0, 300, 100)
        pi_d, p0_d, tau_d = 1e-3, 5e-7, 50
        pressure_d = (pi_d - p0_d) * np.exp(-time_d / tau_d) + p0_d + np.random.normal(0, 1e-8, 100)
        pressure_d_err = np.full_like(pressure_d, 2e-8)
        np.savetxt(decrease_file_default, np.c_[time_d, pressure_d, pressure_d_err], fmt='%.6e',
                   header='time pressure pressure_error', comments='')

    growth_filepath = input(
        f"Enter the path for the pressure growth data file [{growth_file_default}]: ") or growth_file_default
    decrease_filepath = input(
        f"Enter the path for the pressure decrease data file [{decrease_file_default}]: ") or decrease_file_default

    chamber_volume_V = float(input("Enter the chamber volume (V) in liters [100.53]: ") or 100.53)
    pump_speed_S_nominal = float(input("Enter the nominal pump speed in l/s [33.0]: ") or 33.0)

    try:
        f0_growth, f0_growth_err = analyze_pressure_growth(growth_filepath, chamber_volume_V)
        s_eff, s_eff_err, f0_decrease, f0_decrease_err = analyze_pressure_decrease(decrease_filepath, chamber_volume_V,
                                                                                   pump_speed_S_nominal)

        f_comp = np.abs(f0_growth - f0_decrease) / (np.sqrt(f0_growth_err ** 2 + f0_decrease_err ** 2))
        print("\n--- FINAL RESULTS & COMPARISON ---")
        print(f"Flux from Growth Phase (F0): {f0_growth:.4e} ± {f0_growth_err:.2e} mbar·L/s")
        print(f"Flux from Decrease Phase (F0): {f0_decrease:.4e} ± {f0_decrease_err:.2e} mbar·L/s")
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