import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse


def model(q, params, constrained=True):
    if constrained:
        rho1, sigma1, eps1, sigma2, rho3, sigma3, eps3 = params
        rho2 = 30.5  # Constraint
        eps2 = 12.0  # Constraint
    else:
        rho1, sigma1, eps1, rho2, sigma2, eps2, rho3, sigma3, eps3 = params

    rho = [rho1, rho2, rho3]
    sigma = [sigma1, sigma2, sigma3]
    eps = [eps1, eps2, eps3]

    result = 0
    for k in range(3):
        for kp in range(3):
            result += (
                rho[k]
                * rho[kp]
                * sigma[k]
                * sigma[kp]
                * np.exp(-(q**2) * (sigma[k] ** 2 + sigma[kp] ** 2) / 2)
                * np.cos(q * (eps[k] - eps[kp]))
            )

    return q**-2 * result


def objective(params, q, I_exp, constrained=True):
    I_calc = model(q, params, constrained)
    return np.sum((I_calc - I_exp) ** 2)


def fit_saxs(q, I_exp, release_constraints=False):
    initial_guess = [1.71, 3.43, -20.9, 7.79, 1.71, 3.43, 20.9]
    bounds = [
        (0, None),
        (0, None),
        (None, 0),
        (0, None),
        (0, None),
        (0, None),
        (0, None),
    ]

    result = minimize(
        objective,
        initial_guess,
        args=(q, I_exp, True),
        method="L-BFGS-B",
        bounds=bounds,
    )

    params_constrained = result.x

    if release_constraints:
        initial_guess_unconstrained = list(params_constrained)
        initial_guess_unconstrained.insert(3, -1)
        initial_guess_unconstrained.insert(5, 0)

        bounds_unconstrained = [
            (0, None),
            (0, None),
            (None, 0),
            (None, None),
            (0, None),
            (None, None),
            (0, None),
            (0, None),
            (0, None),
        ]

        result = minimize(
            objective,
            initial_guess_unconstrained,
            args=(q, I_exp, False),
            method="L-BFGS-B",
            bounds=bounds_unconstrained,
        )

        return result.x, False
    else:
        return params_constrained, True


def main():
    parser = argparse.ArgumentParser(
        description="Fit SAXS data using a three-shell model."
    )
    parser.add_argument(
        "-f", "--file", required=True, help="Input file with q and I data"
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot the original data and fit"
    )
    parser.add_argument(
        "-r",
        "--release",
        action="store_true",
        help="Release constraints after initial fit",
    )
    parser.add_argument("-o", "--output", help="Output file for fitted function")
    args = parser.parse_args()

    data = np.loadtxt(args.file)
    q, I_exp = data[:, 0], data[:, 1]

    params, constrained = fit_saxs(q, I_exp, args.release)

    print("Fitted parameters:")
    if constrained:
        print("(Constrained fit)")
        print(f"rho1 = {params[0]:.3f}")
        print(f"sigma1 = {params[1]:.3f}")
        print(f"eps1 = {params[2]:.3f}")
        print(f"sigma2 = {params[3]:.3f}")
        print(f"rho3 = {params[4]:.3f}")
        print(f"sigma3 = {params[5]:.3f}")
        print(f"eps3 = {params[6]:.3f}")
        print("rho2 = -1 (constrained)")
        print("eps2 = 0 (constrained)")
    else:
        print("(Unconstrained fit)")
        print(f"rho1 = {params[0]:.3f}")
        print(f"sigma1 = {params[1]:.3f}")
        print(f"eps1 = {params[2]:.3f}")
        print(f"rho2 = {params[3]:.3f}")
        print(f"sigma2 = {params[4]:.3f}")
        print(f"eps2 = {params[5]:.3f}")
        print(f"rho3 = {params[6]:.3f}")
        print(f"sigma3 = {params[7]:.3f}")
        print(f"eps3 = {params[8]:.3f}")

    I_fit = model(q, params, constrained)

    if args.output:
        q_out = np.logspace(np.log10(0.1), np.log10(2.0), num=1000)
        I_out = model(q_out, params, constrained)
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
        plt.title("SAXS Data and Fit")
        plt.show()


if __name__ == "__main__":
    main()
