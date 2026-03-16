#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the PDF of the standard Gaussian N(0,1) and its 1st–3rd derivatives.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
import os

# Standard normal PDF: phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
def gaussian_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

# 1st derivative: phi'(x) = -x * phi(x)
def gaussian_pdf_1st(x):
    return -x * gaussian_pdf(x)

# 2nd derivative: phi''(x) = (x^2 - 1) * phi(x)
def gaussian_pdf_2nd(x):
    return (x**2 - 1) * gaussian_pdf(x)

# 3rd derivative: phi'''(x) = -(x^3 - 3*x) * phi(x)
def gaussian_pdf_3rd(x):
    return -(x**3 - 3 * x) * gaussian_pdf(x)


def main():
    x = np.linspace(-4, 4, 501)

    phi = gaussian_pdf(x)
    d1 = gaussian_pdf_1st(x)
    d2 = gaussian_pdf_2nd(x)
    d3 = gaussian_pdf_3rd(x)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    axes[0, 0].plot(x, phi, "b-", lw=2, label=r"$\phi(x)$")
    axes[0, 0].set_ylabel(r"$\phi(x)$")
    axes[0, 0].set_title("Gaussian PDF")
    axes[0, 0].legend(loc="upper right")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(0, color="k", linewidth=0.5)

    axes[0, 1].plot(x, d1, "green", lw=2, label=r"$\phi'(x) = -x\,\phi(x)$")
    axes[0, 1].set_ylabel(r"$\phi'(x)$")
    axes[0, 1].set_title("1st derivative")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(0, color="k", linewidth=0.5)

    axes[1, 0].plot(x, d2, "orange", lw=2, label=r"$\phi''(x) = (x^2-1)\,\phi(x)$")
    axes[1, 0].set_ylabel(r"$\phi''(x)$")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_title("2nd derivative")
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(0, color="k", linewidth=0.5)

    axes[1, 1].plot(x, d3, "red", lw=2, label=r"$\phi'''(x) = -(x^3-3x)\,\phi(x)$")
    axes[1, 1].set_ylabel(r"$\phi'''(x)$")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_title("3rd derivative")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color="k", linewidth=0.5)

    plt.suptitle("Standard Gaussian N(0,1) and its derivatives", fontsize=12)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), "gaussian_derivatives_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gaussian_and_derivatives.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Saved: {out_path}")

    # Optional: overlay all curves for easier comparison of scales
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(x, phi, "b-", lw=2, label=r"$\phi(x)$")
    ax.plot(x, d1, "green", lw=1.5, alpha=0.9, label=r"$\phi'(x)$")
    ax.plot(x, d2, "orange", lw=1.5, alpha=0.9, label=r"$\phi''(x)$")
    ax.plot(x, d3, "red", lw=1.5, alpha=0.9, label=r"$\phi'''(x)$")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_title("Gaussian and derivatives (overlay)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    overlay_path = os.path.join(out_dir, "gaussian_and_derivatives_overlay.png")
    plt.savefig(overlay_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Saved: {overlay_path}")


if __name__ == "__main__":
    main()
