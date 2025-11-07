import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.ticker import ScalarFormatter, FixedLocator, NullFormatter


FILENAME = 'IAW.txt'
DELIMITER = '\t'
X_COLUMN_INDEX = 0  # First column (Dx)
Y_COLUMN_INDEX = 2  # Third column (Vpp)
Y_ERROR_COLUMN_INDEX = 3  # Fourth column (errVpp)
HEADER_ROWS = 1  # Number of header lines to skip

# ---------------------
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


def load_data(filename):

    x_data = []
    y_data = []
    y_err_data = []


    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        print("Please make sure the file is in the same directory as the script.")
        return None, None, None

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()


            if len(lines) <= HEADER_ROWS:
                print(f"Error: File '{filename}' is empty or has no data rows.")
                return None, None, None


            data_lines = lines[HEADER_ROWS:]

            for line in data_lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(DELIMITER)


                if len(parts) > max(X_COLUMN_INDEX, Y_COLUMN_INDEX, Y_ERROR_COLUMN_INDEX):
                    try:
                        # Get x value
                        x_str = parts[X_COLUMN_INDEX].replace(',', '.')
                        x_data.append(float(x_str))

                        # Get y value
                        y_str = parts[Y_COLUMN_INDEX].replace(',', '.').upper()
                        y_data.append(float(y_str))

                        # Get y_error value
                        y_err_str = parts[Y_ERROR_COLUMN_INDEX].replace(',', '.').upper()
                        y_err_data.append(float(y_err_str))

                    except ValueError as e:
                        print(f"Warning: Skipping line due to conversion error: '{line}' -> {e}")
                else:
                    print(f"Warning: Skipping malformed line: '{line}'")

            if not x_data or not y_data or not y_err_data:
                print("Error: No valid data was read from the file.")
                return None, None, None

            return np.array(x_data), np.array(y_data), np.array(y_err_data)

    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None, None


def plot_data_with_fit(x, y, y_err):

    if x is None or y is None or y_err is None:
        print("Cannot plot, no data loaded.")
        return

    try:

        log_y = np.log(y)
        log_y_err = y_err / y


        weights = 1.0 / (log_y_err ** 2)


        coeffs, cov = np.polyfit(x, log_y, 1, w=weights, cov=True)

        m = coeffs[0]  # slope
        c = coeffs[1]  # intercept


        dm = np.sqrt(cov[0, 0])
        dc = np.sqrt(cov[1, 1])


        L = -1 / m
        dL = (1 / m ** 2) * dm


        log_y_fit = m * x + c
        y_fit = np.exp(log_y_fit)


        residues = log_y - log_y_fit

        print("--- Fit Results ---")
        print(f"Weighted linear fit on log-scale: log(y) = ({m:.4f} \u00B1 {dm:.4f}) * x + ({c:.4f} \u00B1 {dc:.4f})")
        print(f"This corresponds to L = -1/m = ({L:.2f} \u00B1 {dL:.2f}) cm")
        print("-------------------")


        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            sharex=True,
            figsize=(8, 7),
            gridspec_kw={'height_ratios': [3, 1]}
        )


        ax1.errorbar(x, y, yerr=y_err, label='Data', fmt='o', color='blue', ecolor='cornflowerblue', markersize=3,capsize=3,)


        sort_indices = np.argsort(x)
        fit_label = f'Exponential Fit\nL = ({L:.1f} $\pm$ {dL:.1f}) cm'
        ax1.plot(x[sort_indices], y_fit[sort_indices], 'r--', label=fit_label)


        ax1.set_yscale('log')



        ax1.set_ylabel(f'$V_{{pp}}$ (V) - log scale',fontsize='15')
        ax1.set_title(f'Amplitude Damping',fontsize='17')
        ax1.legend(fontsize='17')

        major_ticks = [10, 15, 20, 25, 30]
        ax1.yaxis.set_major_locator(FixedLocator(major_ticks))
        ax1.yaxis.set_major_formatter(ScalarFormatter())


        minor_ticks = np.arange(10, 31, 1)
        ax1.yaxis.set_minor_locator(FixedLocator(minor_ticks))
        ax1.yaxis.set_minor_formatter(NullFormatter())


        ax1.grid(True, which='major', axis='y', ls='--', c='0.8', linewidth=0.6)

        ax1.grid(True, which='minor', axis='y', ls='--', c='0.8', linewidth=0.6)

        ax1.grid(True, which='major', axis='x', ls='--', c='0.8', linewidth=0.6)




        ax2.errorbar(x, residues, yerr=log_y_err, fmt='o', color='blue', ecolor='cornflowerblue', markersize=3,capsize=3,label='Data')


        ax2.axhline(0, color='red', linestyle='--', linewidth=1)


        ax2.set_ylabel('Residuals',fontsize='15')
        ax2.set_xlabel('$\Delta$x (cm)',fontsize='15')
        ax2.grid(True, which="both", ls="--", c='0.7')


        plt.tight_layout()
        plt.savefig('velfit.pdf')

        plt.show()

    except Exception as e:
        print(f"An error occurred during plotting or fitting: {e}")


def main():

    try:
        import matplotlib
        import numpy
    except ImportError:
        print("---")
        print("Warning: 'matplotlib' or 'numpy' not found.")
        print("Please install them to run this script:")
        print("pip install numpy matplotlib")
        print("---")
        return

    x_values, y_values, y_err_values = load_data(FILENAME)
    plot_data_with_fit(x_values, y_values, y_err_values)


if __name__ == "__main__":
    main()