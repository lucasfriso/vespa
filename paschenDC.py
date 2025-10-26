import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar


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

# --- CONFIGURATION ---

DATA_DIRECTORY = '.'

PASCHEN_DATA_FILE = 'paschen_data.txt'
FIT_POINTS = 9

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
    if exponent == -1:
        formatted_string = f"({mantissa_val:.2f} \pm {mantissa_err:.2f})"
    else:
        formatted_string = f"({mantissa_val:.1f} \pm {mantissa_err:.1f}) \cdot 10^{{{exponent}}}"
    return formatted_string

def load_breakdown_data(directory):
    """
    Loads all breakdown curve (I-V) data files from a specified directory.
    It parses filenames to extract filament current and pressure.

    Args:
        directory (str): The path to the directory containing the data files.

    Returns:
        pandas.DataFrame: A single DataFrame containing all the data, or None if no files are found.
    """
    all_data = []

    file_pattern = re.compile(r'(\d+)_(\d+e[+-]\d+)\.txt')

    print(f"Searching for I-V data files in: {os.path.abspath(directory)}")

    for filename in os.listdir(directory):
        match = file_pattern.match(filename)
        if match:
            filepath = os.path.join(directory, filename)
            print(f"Found data file: {filename}")


            filament_current_raw = int(match.group(1))
            pressure_str = match.group(2)


            filament_current = filament_current_raw / 10.0
            pressure = float(pressure_str)

            try:

                df = pd.read_csv(filepath, sep='\s+', decimal=',', header=0)
                df['filament_current'] = filament_current
                df['pressure'] = pressure
                all_data.append(df)
            except Exception as e:
                print(f"Could not read or parse {filename}. Error: {e}")

    if not all_data:
        print("No valid I-V data files found. Please check file names and format.")
        return None

    return pd.concat(all_data, ignore_index=True)


def plot_iv_by_filament_current(data):
    """
    ANALYSIS 1: Creates plots of I-V characteristics, grouped by pressure.
    Each plot shows curves for different filament currents.
    """
    if data is None: return


    min_current = data['current'].min()
    max_current = 1.7
    padding = (max_current - min_current) * 0.1


    if padding < 1e-6:
        padding = max_current * 0.1 if max_current > 1e-6 else 0.1

    y_bottom = min_current - padding
    y_top = max_current + padding


    for pressure, group in data.groupby('pressure'):
        plt.figure(figsize=(10, 7))

        for current, subgroup in group.groupby('filament_current'):


            eb = plt.errorbar(subgroup['voltage'], subgroup['current'],
                              yerr=subgroup['current_error'],
                              fmt='o', markersize=2, capsize=3, label=f'{current} A')


            marker_color = eb[0].get_color()


            for bar in eb[2]:
                bar.set_color(marker_color)
                bar.set_alpha(0.6)


            for cap in eb[1]:
                cap.set_color(marker_color)
                cap.set_alpha(0.6)


        plt.title(f'I-V Characteristics at Constant Pressure: {pressure:.1e} mbar',fontsize='17')
        plt.xlabel('Voltage (V)',fontsize='15')
        plt.ylabel('Discharge Current (A)',fontsize='15')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        legend=plt.legend(title='Filament Current',fontsize='17')
        plt.setp(legend.get_title(), fontsize='17')
        plt.ylim(y_bottom, y_top)
       # plt.yscale('log')

        filename = f'iv_by_current_at_{pressure:.1e}_mbar.pdf'.replace('-', '_')
        plt.savefig(filename)
        #plt.show()
        print(f"Saved plot: {filename}")
        plt.close()


def plot_iv_by_argon_pressure(data):
    """
    ANALYSIS 2: Creates plots of I-V characteristics, grouped by filament current.
    Each plot shows curves for different pressures.
    """
    if data is None: return


    min_current = data['current'].min()
    max_current = 1.7
    padding = (max_current - min_current) * 0.1  # 10% padding


    if padding < 1e-6:
        padding = max_current * 0.1 if max_current > 1e-6 else 0.1

    y_bottom = min_current - padding
    y_top = max_current + padding


    for f_current, group in data.groupby('filament_current'):
        plt.figure(figsize=(10, 7))


        for pressure, subgroup in group.groupby('pressure'):


            eb = plt.errorbar(subgroup['voltage'], subgroup['current'],
                    yerr=subgroup['current_error'],
                         fmt='o', markersize=2, capsize=3, label=f'{pressure:.1e} mbar')


            marker_color = eb[0].get_color()


            for bar in eb[2]:
                bar.set_color(marker_color)
                bar.set_alpha(0.6)


            for cap in eb[1]:
                cap.set_color(marker_color)
                cap.set_alpha(0.6)


        plt.title(f'I-V Characteristics at Constant Filament Current: {f_current} A',fontsize='17')
        plt.xlabel('Voltage (V)',fontsize='15')
        plt.ylabel('Discharge Current (A)',fontsize='15')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        legend=plt.legend(title='Argon Pressure',fontsize='17')
        plt.setp(legend.get_title(), fontsize='17')
        plt.ylim(y_bottom, y_top)
        #plt.yscale('log')

        filename=f'iv_by_current_at_{f_current}_A.pdf'.replace('-', '_')
        plt.savefig(filename)
        #plt.show()
        print(f"Saved plot: {filename}")
        plt.close()


def plot_paschen_curve(filepath):
    """
    ANALYSIS 3: Reads a separate file to plot the Paschen Curve, finds the minimum,
    and performs a quadratic fit around that minimum.
    """
    if not os.path.exists(filepath):
        print(f"\nPaschen curve data file not found: '{filepath}'")
        print(
            "Skipping Paschen curve analysis. To run this, create the file with 'pressure' and 'breakdown_voltage' columns.")
        return

    print(f"\n--- Paschen Curve Analysis ---")
    print(f"Loading data from {filepath}")

    try:
        df = pd.read_csv(filepath, sep='\s+', decimal='.', header=0)
        df.columns = ['pressure', 'pressure_err', 'breakdown_voltage', 'breakdown_voltage_err']

        df = df.sort_values(by='pressure').reset_index(drop=True)
    except Exception as e:
        print(
            f"Could not read or parse {filepath}. Please check the file format and update the parsing logic. Error: {e}")
        return

    plt.figure(figsize=(10, 7))
    plt.errorbar(df['pressure'], df['breakdown_voltage'], yerr=df['breakdown_voltage_err'],
                 fmt='o',color='blue',ecolor='cornflowerblue', markersize=2, capsize=3, label='Experimental Data', zorder=1)

    if not df.empty:

        min_index = df['breakdown_voltage'].idxmin()


        min_pressure_exp = df.loc[min_index, 'pressure']
        min_pressure_err=df.loc[min_index, 'pressure_err']
        min_voltage_exp = df.loc[min_index, 'breakdown_voltage']
        min_voltage_err=df.loc[min_index, 'breakdown_voltage_err']

        min_str=format_sci_pm(min_pressure_exp, min_pressure_err)
        minp_str=format_sci_pm(min_voltage_exp, min_voltage_err)
        fit_label = (f'Experimental min:' +
                     f'\n $p_{{min}}={min_str}$ mbar'+f'\n $V_{{min}}={minp_str}$ V')

        plt.plot(min_pressure_exp, min_voltage_exp,
                 'ro',  # Red circle
                 markersize=15,
                 markerfacecolor='none',  # Hollow
                 markeredgewidth=2,  # Thicker line
                 label=f'Experimental Min: ({min_pressure_exp:.2e} mbar, {min_voltage_exp:.1f} V)',
                 zorder=2)


        fit_data = df.iloc[1:10]
        if len(fit_data) >= 3:

            log_pressure = np.log(fit_data['pressure'])
            voltage = fit_data['breakdown_voltage']


            coeffs = np.polyfit(log_pressure, voltage, 2)
            poly_func = np.poly1d(coeffs)


            log_p_min_fit = -coeffs[1] / (2 * coeffs[0])
            p_min_fit = np.exp(log_p_min_fit)
            v_min_fit = poly_func(log_p_min_fit)


            log_p_smooth = np.linspace(log_pressure.min(), log_pressure.max(), 200)
            v_smooth = poly_func(log_p_smooth)


            print(f"\nFit successful. Calculated minimum at P={p_min_fit:.3e} mbar, V={v_min_fit:.2f} V")
        else:
            print("Not enough data points around the minimum to perform a quadratic fit.")

    plt.title('Paschen Curve for Argon',fontsize='17')
    plt.xlabel('Pressure (mbar)',fontsize='15')
    plt.ylabel('Breakdown Voltage (V)',fontsize='15')
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.legend(fontsize='17')

    filename = 'paschen_curve_with_fit.pdf'
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.show()
    plt.close()


def paschen_func(p, A, B, C):
    """
    Theoretical Paschen's Law function.
    V = (B * p) / (ln(A * p) - C)
    Here, p is pressure (since distance d is constant).
    A, B are fit parameters related to the gas properties.
    C is a fit parameter related to the secondary emission coefficient gamma.
    C = ln(ln(1 + 1/gamma))
    """

    arg = A * p
    return (B * p) / (np.log(arg) - C)


def plot_paschen_curve_full_fit(filepath):
    """
    ANALYSIS 3 (REVISED): Reads Paschen data, performs a non-linear fit
    using the theoretical Paschen's Law, and finds the minimum.
    """
    if not os.path.exists(filepath):
        print(f"\nPaschen curve data file not found: '{filepath}'")
        print("Skipping Paschen curve analysis.")
        return

    print(f"\n--- Paschen Curve Analysis (Full Theoretical Fit) ---")
    print(f"Loading data from {filepath}")

    try:
        df = pd.read_csv(filepath, sep='\s+', decimal='.', header=0)
        df.columns = ['pressure', 'pressure_err', 'breakdown_voltage', 'breakdown_voltage_err']
        df = df.sort_values(by='pressure').reset_index(drop=True)
    except Exception as e:
        print(f"Could not read or parse {filepath}. Error: {e}")
        return

    plt.figure(figsize=(10, 7))

    plt.errorbar(df['pressure'], df['breakdown_voltage'], xerr=df['pressure_err'], yerr=df['breakdown_voltage_err'],
                 fmt='o', markersize=4, capsize=3, label='Experimental Data', zorder=1)

    if df.empty:
        print("No data to plot.")
        plt.close()
        return


    min_index = df['breakdown_voltage'].idxmin()
    min_pressure_exp = df.loc[min_index, 'pressure']
    min_voltage_exp = df.loc[min_index, 'breakdown_voltage']
    plt.plot(min_pressure_exp, min_voltage_exp, 'r*', markersize=12,
             label=f'Experimental Min: ({min_pressure_exp:.2e} mbar, {min_voltage_exp:.1f} V)', zorder=2)


    try:

        initial_guesses = [10.0, 200.0, 1.0]


        popt, pcov = curve_fit(
            paschen_func,
            df['pressure'],
            df['breakdown_voltage'],
            p0=initial_guesses,
            sigma=df['breakdown_voltage_err'],
            absolute_sigma=True,
            maxfev=10000
        )

        A_fit, B_fit, C_fit = popt
        print("\n--- Fit Results ---")
        print(f"Fit converged.")
        print(f"Fitted A: {A_fit:.3f}")
        print(f"Fitted B: {B_fit:.3f}")
        print(f"Fitted C: {C_fit:.3f}")


        p_smooth = np.logspace(
            np.log10(df['pressure'].min()),
            np.log10(df['pressure'].max()),
            500  # 500 points for a smooth line
        )
        v_smooth = paschen_func(p_smooth, A_fit, B_fit, C_fit)

        plt.plot(p_smooth, v_smooth, 'g-',
                 label=f'Paschen\'s Law Fit', zorder=3, linewidth=2)


        fit_func_to_minimize = lambda p: paschen_func(p, A_fit, B_fit, C_fit)


        fit_min = minimize_scalar(
            fit_func_to_minimize,
            bounds=(df['pressure'].min(), df['pressure'].max()),
            method='bounded'
        )

        p_min_fit = fit_min.x
        v_min_fit = fit_min.fun

        print(f"Calculated Fit Minimum at P={p_min_fit:.3e} mbar, V={v_min_fit:.2f} V")

        plt.plot(p_min_fit, v_min_fit, 'gX', markersize=10,
                 label=f'Fitted Min: ({p_min_fit:.2e} mbar, {v_min_fit:.1f} V)', zorder=4)

    except RuntimeError as e:
        print(f"\n--- Fit Failed ---")
        print(f"Error: {e}")
        print("Could not converge on a fit. The plot will only show experimental data.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during fitting: {e}")

    plt.title('Paschen Curve for Argon with Theoretical Fit')
    plt.xlabel('Pressure (mbar)')
    plt.ylabel('Breakdown Voltage (V)')
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.legend(fontsize='12')

    filename = 'paschen_curve_with_full_fit.pdf'
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.show()
    plt.close()


if __name__ == '__main__':

    all_breakdown_data = load_breakdown_data(DATA_DIRECTORY)

    if all_breakdown_data is not None:
        print("\n--- Starting I-V Curve Analysis ---")

        plot_iv_by_filament_current(all_breakdown_data)


        plot_iv_by_argon_pressure(all_breakdown_data)


    plot_paschen_curve(PASCHEN_DATA_FILE)

    print("\n--- Analysis Complete ---")
