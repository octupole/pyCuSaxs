#!/usr/bin/env python3
"""Compare DE and CMA-ES fit results."""

import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt('pycusaxs/fit/data/test_data.dat')
q_all = data[:, 0]
y_all = data[:, 1]

# Apply q-range filter
mask = (q_all >= 0.02) & (q_all <= 0.6)
q = q_all[mask]
y = y_all[mask]

# Load fits
fit_de = np.loadtxt('bicelle_fit_de_result.dat')
fit_cma = np.loadtxt('bicelle_fit_cma_result.dat')

# Create comparison plot
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top: log-log plot
ax1 = axes[0]
ax1.loglog(q, y, 'ko', ms=4, alpha=0.5, label='Data')
ax1.loglog(fit_de[:, 0], fit_de[:, 1], 'r-', lw=2, label='DE fit')
ax1.loglog(fit_cma[:, 0], fit_cma[:, 1], 'b-', lw=2, label='CMA-ES fit')
ax1.set_ylabel('I(q)')
ax1.legend()
ax1.set_title('Bicelle Fit Comparison: DE vs CMA-ES')
ax1.grid(True, alpha=0.3)

# Bottom: residuals
ax2 = axes[1]
# Interpolate fit values at data q points
y_de_interp = np.interp(q, fit_de[:, 0], fit_de[:, 1])
y_cma_interp = np.interp(q, fit_cma[:, 0], fit_cma[:, 1])

residuals_de = (y_de_interp - y) / y * 100
residuals_cma = (y_cma_interp - y) / y * 100

ax2.semilogx(q, residuals_de, 'ro', ms=3, alpha=0.5, label='DE residuals')
ax2.semilogx(q, residuals_cma, 'bo', ms=3, alpha=0.5, label='CMA-ES residuals')
ax2.axhline(0, color='k', ls='--', lw=1)
ax2.set_xlabel('q (1/Å)')
ax2.set_ylabel('Relative residuals (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fit_comparison.png', dpi=150)
print('Saved comparison plot to fit_comparison.png')

# Print statistics
print('\n' + '='*60)
print('FIT COMPARISON SUMMARY')
print('='*60)

# Extract costs from file headers
with open('bicelle_fit_de_result.dat', 'r') as f:
    for line in f:
        if 'Cost (local)' in line:
            cost_de = float(line.split('=')[1])

with open('bicelle_fit_cma_result.dat', 'r') as f:
    for line in f:
        if 'Cost (local)' in line:
            cost_cma = float(line.split('=')[1])

print(f'DE cost:     {cost_de:.6f}')
print(f'CMA-ES cost: {cost_cma:.6f}')
print(f'Improvement: {(cost_de - cost_cma)/cost_de * 100:.2f}%')

print('\nRMSE (Root Mean Square Error):')
rmse_de = np.sqrt(np.mean(residuals_de**2))
rmse_cma = np.sqrt(np.mean(residuals_cma**2))
print(f'DE:     {rmse_de:.3f}%')
print(f'CMA-ES: {rmse_cma:.3f}%')

print('\nMax absolute residual:')
print(f'DE:     {np.max(np.abs(residuals_de)):.3f}%')
print(f'CMA-ES: {np.max(np.abs(residuals_cma)):.3f}%')
print('='*60)
