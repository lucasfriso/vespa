import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---

DATA_DIRECTORY = '.'

PASCHEN_DATA_FILE = 'paschen_data.txt'
FIT_POINTS = 3

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
    # Regex to parse filenames like '61_2e-03.txt'
    # It captures filament current (e.g., 61) and pressure (e.g., 2e-03)
    file_pattern = re.compile(r'(\d+)_(\d+e[+-]\d+)\.txt')

    print(f"Searching for I-V data files in: {os.path.abspath(directory)}")

    for filename in os.listdir(directory):
        match = file_pattern.match(filename)
        if match:
            filepath = os.path.join(directory, filename)
            print(f"Found data file: {filename}")


            filament_current_raw = int(match.group(1))
            pressure_str = match.group(2)

            # Convert to float values. Assuming filename '62' means 6.2A.
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


    for pressure, group in data.groupby('pressure'):
        plt.figure(figsize=(10, 7))


        for current, subgroup in group.groupby('filament_current'):
            plt.errorbar(subgroup['voltage'], subgroup['current'],
                         xerr=subgroup['voltage_error'], yerr=subgroup['current_error'],
                         fmt='o', markersize=2, capsize=3, label=f'{current} A')

        plt.title(f'I-V Characteristics at Constant Pressure: {pressure:.1e} mbar')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Discharge Current (A)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title='Filament Current')
       # plt.yscale('log')

        filename = f'iv_by_current_at_{pressure:.1e}_mbar.png'.replace('-', '_')
        plt.savefig(filename)
        plt.show()
        print(f"Saved plot: {filename}")
        plt.close()


def plot_iv_by_argon_pressure(data):
    """
    ANALYSIS 2: Creates plots of I-V characteristics, grouped by filament current.
    Each plot shows curves for different pressures.
    """
    if data is None: return


    for f_current, group in data.groupby('filament_current'):
        plt.figure(figsize=(10, 7))


        for pressure, subgroup in group.groupby('pressure'):
            plt.errorbar(subgroup['voltage'], subgroup['current'],
                         xerr=subgroup['voltage_error'], yerr=subgroup['current_error'],
                         fmt='o', markersize=2, capsize=3, label=f'{pressure:.1e} mbar')

        plt.title(f'I-V Characteristics at Constant Filament Current: {f_current} A')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Discharge Current (A)')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(title='Argon Pressure')
        #plt.yscale('log')

        filename = f'iv_by_pressure_at_{f_current}_A.png'.replace('.', '_')
        plt.savefig(filename)
        plt.show()
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
    plt.errorbar(df['pressure'], df['breakdown_voltage'], xerr=df['pressure_err'], yerr=df['breakdown_voltage_err'],
                 fmt='o', markersize=4, capsize=3, label='Experimental Data', zorder=1)

    if not df.empty:

        min_index = df['breakdown_voltage'].idxmin()


        min_pressure_exp = df.loc[min_index, 'pressure']
        min_voltage_exp = df.loc[min_index, 'breakdown_voltage']
        plt.plot(min_pressure_exp, min_voltage_exp, 'r*', markersize=12,
                 label=f'Experimental Min: ({min_pressure_exp:.2e} mbar, {min_voltage_exp:.1f} V)', zorder=2)


        half_fit = FIT_POINTS // 2
        start_index = max(0, min_index - half_fit)
        end_index = min(len(df), min_index + half_fit + 1)
        fit_data = df.iloc[start_index:end_index]

        if len(fit_data) >= 3:

            log_pressure = np.log(fit_data['pressure'])
            voltage = fit_data['breakdown_voltage']


            coeffs = np.polyfit(log_pressure, voltage, 2)
            poly_func = np.poly1d(coeffs)


            log_p_min_fit = -coeffs[1] / (2 * coeffs[0])
            p_min_fit = np.exp(log_p_min_fit)
            v_min_fit = poly_func(log_p_min_fit)

            # 4. Generate smooth curve for plotting the fit
            log_p_smooth = np.linspace(log_pressure.min(), log_pressure.max(), 200)
            v_smooth = poly_func(log_p_smooth)

            plt.plot(np.exp(log_p_smooth), v_smooth, 'g-',
                     label=f'Quadratic Fit', zorder=3, linewidth=2)
            plt.plot(p_min_fit, v_min_fit, 'gX', markersize=10,
                     label=f'Fitted Min: ({p_min_fit:.2e} mbar, {v_min_fit:.1f} V)', zorder=4)
            print(f"\nFit successful. Calculated minimum at P={p_min_fit:.3e} mbar, V={v_min_fit:.2f} V")
        else:
            print("Not enough data points around the minimum to perform a quadratic fit.")

    plt.title('Paschen Curve for Argon with Quadratic Fit at Minimum')
    plt.xlabel('Pressure (mbar)')
    plt.ylabel('Breakdown Voltage (V)')
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()

    filename = 'paschen_curve_with_fit.png'
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
