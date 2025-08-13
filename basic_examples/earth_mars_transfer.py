"""
deterministic_earth_mars_transfer.py

Deterministic Earth-Mars transfer baseline:
- Loads ephemeris using Skyfield
- Computes heliocentric states for Earth & Mars at given dates
- Solves Lambert's problem for the specified time-of-flight
- Computes Delta-Vs (simple impulsive model)
- Plots heliocentric geometry

Dependencies:
    numpy, scipy, matplotlib, skyfield, astropy

Notes:
- Skyfield will try to download planetary BSP files automatically.
- Adjust departure/arrival dates in the `if __name__ == "__main__":` block.
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from skyfield.api import load
from datetime import datetime
from math import pi

# Gravitational parameter of Sun [km^3/s^2]
MU_SUN = 1.32712440018e11

# -----------------------
# Utility / Ephemeris
# -----------------------
def load_ephemeris(bsp_name='de421.bsp'):
    """
    Load planetary ephemeris using skyfield. If not present skyfield will download.
    Returns timescale `ts` and loaded planetary kernel.
    """
    ts = load.timescale()
    planets = load(bsp_name)  # skyfield will fetch if not already cached
    return ts, planets

def get_heliocentric_state(body, t, sun):
    """
    Return heliocentric position (km) and velocity (km/s) of `body` at skyfield time `t`.
    `body` is a skyfield planet object (e.g., planets['earth'])
    `sun` is planets['sun'] object for subtraction reference.
    """
    # skyfield .at(t).position.km gives position in its frame (e.g., solar-system barycentric)
    # We'll compute position relative to Sun:
    pos = (body.at(t) - sun.at(t)).position.km  # shape (3,)
    vel = (body.at(t) - sun.at(t)).velocity.km_per_s  # shape (3,)
    return pos, vel

# -----------------------
# Stumpff helpers
# -----------------------
def stumpff_c2(z):
    if z > 1e-8:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < -1e-8:
        return (np.cosh(np.sqrt(-z)) - 1) / -z
    else:
        # series expansion near z=0
        return 0.5 - z / 24.0 + z**2 / 720.0

def stumpff_c3(z):
    if z > 1e-8:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (z**1.5)
    elif z < -1e-8:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / ((-z)**1.5)
    else:
        return 1.0/6.0 + z / 120.0 + z**2 / 5040.0

# -----------------------
# Lambert solver (single-rev, prograde only baseline)
# -----------------------
def lambert_universal(r1, r2, tof, mu=MU_SUN, tol=1e-6, maxiter=200):
    """
    A reasonable universal-variable Lambert solver for initial use.
    Input:
        r1, r2 : position vectors km (numpy arrays)
        tof    : time-of-flight in seconds (scalar)
    Output:
        v1, v2 : transfer velocities at r1 and r2 (km/s)
    Note: This implementation handles the typical elliptic transfer case.
    """
    r1_norm = norm(r1)
    r2_norm = norm(r2)
    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    # numerical safety
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)
    dnu = np.arccos(cos_dnu)

    # choose shorter arc by default (prograde assumption)
    if np.cross(r1, r2)[2] < 0:
        dnu = 2*pi - dnu

    A = np.sin(dnu) * np.sqrt(r1_norm * r2_norm / (1.0 - cos_dnu))
    if A == 0:
        raise ValueError("A = 0 in Lambert setup (collinear vectors?)")

    # initial z guess
    z = 0.0
    # Newton iteration on z to match tof
    for i in range(maxiter):
        c2 = stumpff_c2(z)
        c3 = stumpff_c3(z)

        y = r1_norm + r2_norm + A * (z * c3 - 1.0) / np.sqrt(max(c2, 1e-12))
        if A > 0 and y < 0:
            # push z positive
            z += 0.1
            continue

        # avoid negative sqrt
        chi = np.sqrt(max(y / max(c2, 1e-12), 0.0))
        tof_z = (chi**3 * c3 + A * np.sqrt(max(y, 0.0))) / np.sqrt(mu)

        # derivative dt/dz (approx.)
        if abs(z) < 1e-8:
            dto_dz = (np.sqrt(2)/40.0) * y**1.5 + (A/8.0) * (np.sqrt(y) + A / np.sqrt(2.0 * max(y,1e-12)))
        else:
            # finite-difference fallback if analytic derivative is unstable
            dz = 1e-6 * max(1.0, abs(z))
            c2_p = stumpff_c2(z + dz)
            c3_p = stumpff_c3(z + dz)
            y_p = r1_norm + r2_norm + A * ((z + dz) * c3_p - 1.0) / np.sqrt(max(c2_p,1e-12))
            chi_p = np.sqrt(max(y_p / max(c2_p,1e-12), 0.0))
            tof_p = (chi_p**3 * c3_p + A * np.sqrt(max(y_p,0.0))) / np.sqrt(mu)
            dto_dz = (tof_p - tof_z) / dz

        # Newton step
        fval = tof_z - tof
        # avoid huge steps
        dz = -fval / dto_dz
        if abs(dz) > 1.0:
            dz = np.sign(dz) * 1.0
        z += dz

        if abs(fval) < tol:
            break
    else:
        print("lambert_universal: Warning — max iterations reached, residual:", fval)

    # final y, f, g functions
    c2 = stumpff_c2(z)
    c3 = stumpff_c3(z)
    y = r1_norm + r2_norm + A * (z * c3 - 1.0) / np.sqrt(max(c2,1e-12))
    f = 1.0 - y / r1_norm
    g = A * np.sqrt(max(y,0.0) / mu)
    gdot = 1.0 - y / r2_norm

    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g

    return v1, v2

# -----------------------
# Delta-V estimation (impulsive)
# -----------------------
def compute_dv(v_transfer_dep, v_body_dep, v_transfer_arr, v_body_arr):
    """
    Simple impulsive delta-V estimates:
      DV_dep = ||v_transfer_dep - v_body_dep||
      DV_arr = ||v_body_arr - v_transfer_arr||
    Returns (dv_dep, dv_arr, total_dv)
    Note: Real mission modelling would include parking orbit insertion/extraction.
    """
    dv_dep = norm(v_transfer_dep - v_body_dep)
    dv_arr = norm(v_body_arr - v_transfer_arr)
    return dv_dep, dv_arr, dv_dep + dv_arr

# -----------------------
# Plotting
# -----------------------
def plot_geometry(r1, r2, r_points=None, show=True, title="Heliocentric Geometry"):
    """
    Simple 2D plot of heliocentric positions and transfer chord.
    r_points: optional Nx3 array of transfer points to plot trajectory
    """
    plt.figure(figsize=(8,8))
    plt.plot(0,0,'yo',label='Sun')
    plt.plot(r1[0], r1[1], 'bo', label='Earth (dep)')
    plt.plot(r2[0], r2[1], 'ro', label='Mars (arr)')
    plt.plot([r1[0], r2[0]], [r1[1], r2[1]], 'k--', label='Transfer chord')

    if r_points is not None:
        plt.plot(r_points[:,0], r_points[:,1], '-', label='Transfer (approx)')

    # Plot circular reference orbits (optional)
    rmax = max(norm(r1), norm(r2)) * 1.2
    circle = np.linspace(0, 2*pi, 200)
    plt.plot(np.cos(circle)*norm(r1), np.sin(circle)*norm(r1), ':', alpha=0.4)
    plt.plot(np.cos(circle)*norm(r2), np.sin(circle)*norm(r2), ':', alpha=0.4)

    plt.gca().set_aspect('equal', 'box')
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.title(title)
    plt.legend()
    if show:
        plt.show()

# -----------------------
# Example runner
# -----------------------
def run_transfer(dep_date_tuple, arr_date_tuple, bsp='de421.bsp', plot=True):
    """
    High-level convenience function.
    dep_date_tuple, arr_date_tuple : (year, month, day) or (year, month, day, hour, minute, second)
    bsp: name of BSP file (skyfield)
    Returns a dictionary with results
    """
    ts, planets = load_ephemeris(bsp)
    earth = planets['earth']
    mars = planets['mars']
    sun = planets['sun']

    # create skyfield times
    ts_dep = ts.utc(*dep_date_tuple)
    ts_arr = ts.utc(*arr_date_tuple)

    # Time of flight (seconds)
    tof_seconds = (ts_arr.tt - ts_dep.tt) * 86400.0

    # get heliocentric states
    r1, v_earth = get_heliocentric_state(earth, ts_dep, sun)
    r2, v_mars  = get_heliocentric_state(mars, ts_arr, sun)

    # Solve Lambert
    v_transfer_dep, v_transfer_arr = lambert_universal(r1, r2, tof_seconds, mu=MU_SUN)

    # compute delta-v
    dv_dep, dv_arr, dv_total = compute_dv(v_transfer_dep, v_earth, v_transfer_arr, v_mars)

    # optional simple transfer sampling (straight chord interpolation for plotting)
    npts = 80
    r_points = np.linspace(r1, r2, npts)

    results = {
        'r1': r1, 'r2': r2,
        'v_earth': v_earth, 'v_mars': v_mars,
        'v_transfer_dep': v_transfer_dep, 'v_transfer_arr': v_transfer_arr,
        'dv_dep': dv_dep, 'dv_arr': dv_arr, 'dv_total': dv_total,
        'tof_s': tof_seconds,
        'r_points': r_points,
        'dep_time': dep_date_tuple, 'arr_time': arr_date_tuple
    }

    # Print summary
    print("Departure (UTC):", dep_date_tuple)
    print("Arrival   (UTC):", arr_date_tuple)
    print(f"Time-of-flight: {tof_seconds/86400.0:.2f} days ({tof_seconds:.1f} s)")
    print("||r1|| = {:.3e} km, ||r2|| = {:.3e} km".format(norm(r1), norm(r2)))
    print("||v_earth|| = {:.6f} km/s".format(norm(v_earth)))
    print("||v_mars || = {:.6f} km/s".format(norm(v_mars)))
    print("||v_transfer_dep|| = {:.6f} km/s, ||v_transfer_arr|| = {:.6f} km/s".format(norm(v_transfer_dep), norm(v_transfer_arr)))
    print("Delta-V (dep): {:.6f} km/s, Delta-V (arr): {:.6f} km/s, Total ΔV: {:.6f} km/s".format(dv_dep, dv_arr, dv_total))

    if plot:
        plot_geometry(r1, r2, r_points, show=True, title=f"Earth->Mars transfer; TOF {tof_seconds/86400.0:.1f} d")

    return results

# -----------------------
# CLI / Example execution
# -----------------------
if __name__ == "__main__":
    # Example dates (edit these to test various launch windows)
    # Departure ~ Nov 7, 2024; arrival ~ Jun 5, 2025 ( ~ 210 - 260 days typical transfer)
    departure = (2024, 11, 7, 0, 0, 0)
    arrival   = (2025, 6, 5, 0, 0, 0)

    # Run the deterministic baseline
    res = run_transfer(departure, arrival, bsp='de421.bsp', plot=True)
