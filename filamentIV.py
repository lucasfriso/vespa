import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def theoretical_current(v, consts):
    """
    Calculates the theoretical current (I_F) as a function of voltage (V_F).
    This is the inverted form of the user-provided equation.
    I_F = K * V_F^(7/13)
    """

    C = consts['C']
    L = consts['L']
    epsilon = consts['epsilon']
    sigma = consts['sigma']
    r = consts['r']
    pi = np.pi


    numerator = (2 ** 3 * pi ** 13 * epsilon ** 3 * sigma ** 3 * r ** 23)
    denominator = (C ** 10 * L ** 7)


    k_factor = (numerator / denominator) ** (1 / 13)

    return k_factor * v ** (7 / 13)


def calculate_temperature(current, current_error, consts):
    """
    Calculates temperature (T) and its propagated error from current (I).
    T = K_T * I^(5/7)
    """

    C = consts['C']
    epsilon = consts['epsilon']
    sigma = consts['sigma']
    r = consts['r']
    pi = np.pi


    k_t_factor = (C / (2 * pi ** 2 * epsilon * sigma * r ** 3)) ** (5 / 14)


    temperature = k_t_factor * current ** (5 / 7)


    relative_current_error = np.divide(current_error, current, out=np.zeros_like(current_error), where=current != 0)
    temperature_error = temperature * (5 / 7) * relative_current_error

    return temperature, temperature_error




def plot_iv_curve_with_residuals(data, consts):
    """
    Plots experimental I-V data and the theoretical curve.
    """
    V = data['voltage']
    V_err = data['voltage_error']
    I = data['current']
    I_err = data['current_error']


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})


    ax1.errorbar(V, I, yerr=I_err, xerr=V_err, fmt='o', color='blue', ecolor='cornflowerblue',
                 capsize=3, label='Experimental Data', markersize=3)

    v_theory = np.linspace(min(V) * 0.9, max(V) * 1.1, 65)
    i_theory_line = theoretical_current(v_theory, consts)

    ax1.plot(v_theory, i_theory_line, color='red', linestyle='--', label='Theoretical Model')

    ax1.set_title('Current vs. Voltage (I-V) Characteristic',fontsize='21')
    ax1.set_ylabel('Current (A)',fontsize='21')
    ax1.grid(True, which="both", ls="--")
    ax1.legend(fontsize='19')

    i_theory_points = theoretical_current(V, consts)
    residuals = I - i_theory_points


    residuals_err = np.sqrt(I_err ** 2 + theoretical_current(V_err, consts)**2)

    ax2.errorbar(V, residuals, yerr=residuals_err, fmt='o', color='blue', ecolor='cornflowerblue',
                 capsize=3, markersize=3)
    ax2.axhline(0, color='red', linestyle='--', label='Zero Line')

    ax2.set_xlabel('Voltage (V)',fontsize='21')
    ax2.set_ylabel('Residuals (A)',fontsize='21')
    ax2.grid(True, which="both", ls="--")




    #plt.legend(fontsize='17',position='bottomleft')
    plt.savefig('filamentIV.pdf', dpi=300)
    plt.show()


def plot_temperature_vs_current(data):
    """
    Plots the calculated temperature as a function of experimental current.
    """
    I = data['current']
    I_err = data['current_error']
    T = data['temperature']
    T_err = data['temperature_error']

    plt.figure(figsize=(10, 7))

    plt.errorbar(I, T, yerr=T_err, fmt='o', color='blue', ecolor='cornflowerblue', capsize=3,
                 label='Calculated Temperature', markersize=3)
    plt.plot(I,T,linestyle='--', color='red', label='Theoretical Model')

    plt.title('Filament Temperature vs. Current',fontsize='21')
    plt.xlabel('Current (A)',fontsize='21')
    plt.ylabel('Temperature (K)',fontsize='21')
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize='19')
    plt.savefig('TvI.pdf', dpi=300)
    plt.show()


# --- 3. MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    print("--- I-V Curve and Temperature Analysis ---")

    constants = {
        'C': 6.2e-11,
        'L': 0.1,
        'r': 1.19e-04,
        'epsilon': 0.3,
        'sigma': 5.67e-8
    }

    sample_file = 'sample_iv_data.txt'
    if not os.path.exists(sample_file):
        print(f"Creating a sample data file: {sample_file}")
        v_sample = np.linspace(1, 10, 20)
        i_sample = 0.05 * v_sample ** (7 / 13) + np.random.normal(0, 0.005, 20)
        v_err_sample = np.full_like(v_sample, 0.1)
        i_err_sample = np.full_like(i_sample, 0.008)
        header = "voltage voltage_error current current_error"
        np.savetxt(sample_file, np.c_[v_sample, v_err_sample, i_sample, i_err_sample], fmt='%.6e', header=header,
                   comments='')

    filepath = input(f"Enter the path for your data file [{sample_file}]: ") or sample_file

    print("\nCurrent physical constants (in SI units):")
    for key, val in constants.items():
        print(f"  {key}: {val}")

    if input("Do you want to change any of these values? (y/n) [n]: ").lower() == 'y':
        while True:
            key_to_change = input("Enter the constant to change (or 'done' to finish): ").strip()
            if key_to_change.lower() == 'done':
                break
            if key_to_change in constants:
                try:
                    new_value = float(input(f"Enter the new value for {key_to_change}: "))
                    constants[key_to_change] = new_value
                    print(f"Updated {key_to_change} to {new_value}")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            else:
                print(f"Error: '{key_to_change}' is not a valid constant name.")

    try:
        iv_data = pd.read_csv(filepath, comment='#', header=0, delim_whitespace=True,decimal=',')


        plot_iv_curve_with_residuals(iv_data, constants)


        T, T_err = calculate_temperature(iv_data['current'], iv_data['current_error'], constants)
        iv_data['temperature'] = T
        iv_data['temperature_error'] = T_err


        plot_temperature_vs_current(iv_data)

        print("\nAnalysis complete. Two plots have been generated.")

    except FileNotFoundError:
        print(f"\nError: The file '{filepath}' was not found. Please check the path and try again.")
    except KeyError as e:
        print(
            f"\nError: A required column is missing from your data file: {e}. Please ensure the file has the columns: 'voltage', 'voltage_error', 'current', and 'current_error'.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis: {e}")

