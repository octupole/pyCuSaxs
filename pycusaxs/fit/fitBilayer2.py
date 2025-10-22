import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse


def model_two_layer(q, params):
    rho1, sigma1, eps1, rho2, sigma2, eps2 = params

    rho = [rho1, rho2]
    sigma = [sigma1, sigma2]
    eps = [eps1, eps2]

    result = 0
    for k in range(2):
        for kp in range(2):
            result += (
                rho[k]
                * rho[kp]
                * sigma[k]
                * sigma[kp]
                * np.exp(-(q**2) * (sigma[k] ** 2 + sigma[kp] ** 2) / 2)
                * np.cos(q * (eps[k] - eps[kp]))
            )

    return q**-2 * result


def objective_two_layer(params, q, I_exp):
    I_calc = model_two_layer(q, params)
    return np.sum((I_calc - I_exp) ** 2)


def fit_saxs_two_layer(q, I_exp):
    # Initial guess: [rho1, sigma1, eps1, rho2, sigma2, eps2]
    initial_guess = [1.0, 3.0, -20.0, 1.0, 3.0, 20.0]

    # Bounds to ensure physically meaningful results
    bounds = [(0, None), (0, None), (None, 0), (0, None), (0, None), (0, None)]

    result = minimize(
        objective_two_layer,
        initial_guess,
        args=(q, I_exp),
        method="L-BFGS-B",
        bounds=bounds,
    )

    return result.x


def main():
    parser = argparse.ArgumentParser(
        description="Fit SAXS data using a two-layer model."
    )
    parser.add_argument(
        "-f", "--file", required=True, help="Input file with q and I data"
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot the original data and fit"
    )
    parser.add_argument("-o", "--output", help="Output file for fitted function")
    args = parser.parse_args()

    data = np.loadtxt(args.file)
    q, I_exp = data[:, 0], data[:, 1]

    params = fit_saxs_two_layer(q, I_exp)

    print("Fitted parameters:")
    print(f"rho1 = {params[0]:.3f}")
    print(f"sigma1 = {params[1]:.3f}")
    print(f"eps1 = {params[2]:.3f}")
    print(f"rho2 = {params[3]:.3f}")
    print(f"sigma2 = {params[4]:.3f}")
    print(f"eps2 = {params[5]:.3f}")

    # Calculate bilayer thickness
    d_hh = abs(params[5] - params[2])
    print(f"Bilayer thickness (d_hh) = {d_hh:.3f} Å")

    I_fit = model_two_layer(q, params)

    if args.output:
        # Generate q values from 0.1 to 2.0 A^-1
        q_out = np.logspace(np.log10(0.1), np.log10(2.0), num=1000)
        I_out = model_two_layer(q_out, params)
        np.savetxt(
            args.output,
            np.column_stack((q_out, I_out)),
            delimiter="\t",
            header="q\tI(q)_fit",
            comments="",
        )
        print(f"Fitted function written to {args.output}")

    if args.plot:
        plt.figure(figsize=(10, 6))
        plt.loglog(q, I_exp, "o", label="Experimental")
        plt.loglog(q, I_fit, "-", label="Fit")
        plt.xlabel("q (Å^-1)")
        plt.ylabel("I(q)")
        plt.legend()
        plt.title("SAXS Data and Fit (Two-Layer Model)")
        plt.show()


if __name__ == "__main__":
    main()
