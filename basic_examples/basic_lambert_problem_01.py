import numpy as np
from numpy.linalg import norm
from scipy.optimize import newton
import matplotlib.pyplot as plt
from skyfield.api import load

# Constants
mu_sun = 1.32712440018e11  # km^3/s^2

# Load planetary ephemeris
ts = load.timescale()
planets = load('de421.bsp')
earth, mars, sun = planets['earth'], planets['mars'], planets['sun']

#least amount of energy.
# Define departure and arrival times
departure_time = ts.utc(2024, 11, 7)  # Approx. Earth-Mars Hohmann window
print("Departure time:", departure_time)
arrival_time = ts.utc(2025, 6, 5)    # ~260 days later

tof = (arrival_time - departure_time)*24 * 3600  # Convert to seconds

# Get position vectors from Earth and Mars w.r.t. the Sun
r1 = (earth.at(departure_time) - sun.at(departure_time)).position.km
r2 = (mars.at(arrival_time) - sun.at(arrival_time)).position.km
print("Position vector from Sun to Earth (r1):", r1, "[km]")
print("Position vector from Sun to Mars (r2):", r2, "[km]")
print("||r1|| =", np.linalg.norm(r1), "km")
print("||r2|| =", np.linalg.norm(r2), "km")
# Stumpff functions  are a set of special functions used in orbital mechanics,
# especially when solving Kepler’s equation in the universal variable formulation of the two-body problem.
# They allow for a unified treatment of elliptic, parabolic, and hyperbolic orbits in a single framework.

def stumpff_c2(z):
    """
    Computes the Stumpff function C2(z).

    Args:
        z (float): The universal variable.

    Returns:
        float: The value of the Stumpff function C2.
    """
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    elif z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / -z
    else:
        return 0.5

def stumpff_c3(z):
    """
    Computes the Stumpff function C3(z).

    Args:
        z (float): The universal variable.

    Returns:
        float: The value of the Stumpff function C3.
    """
    if z > 0:
        return (np.sqrt(z) - np.sin(np.sqrt(z))) / (z**1.5)
    elif z < 0:
        return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / ((-z)**1.5)
    else:
        return 1/6

def lambert_solver(r1, r2, tof, mu, tol=1e-6, max_iter=100):
    """
    Solves Lambert's problem to find the initial and final velocities.

    Args:
        r1 (np.ndarray): Initial position vector (3x1).
        r2 (np.ndarray): Final position vector (3x1).
        tof (float): Time of flight seconds.
        mu (float): Gravitational parameter of the central body.
        tol (float, optional): Tolerance for the solver. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.

    Returns:
        tuple: A tuple containing the initial velocity vector (v1) and the
               final velocity vector (v2).
    """
    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)

    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    dnu = np.arccos(cos_dnu)

    # Determine the direction of motion
    if np.cross(r1, r2)[2] < 0:
        dnu = 2 * np.pi - dnu

    A = np.sin(dnu) * np.sqrt(r1_norm * r2_norm / (1 - cos_dnu))

    # Initial guess for z
    z = 0.0
    dt = 0.0
    iterations = 0

    while abs(dt - tof) > tol and iterations < max_iter:
        c2 = stumpff_c2(z)
        c3 = stumpff_c3(z)

        y = r1_norm + r2_norm + A * (z * c3 - 1) / np.sqrt(c2)
        
        if A > 0 and y < 0:
            # Adjust z to ensure y is positive
            z = z + 0.1 # Simple adjustment, a more robust method might be needed
            continue

        chi = np.sqrt(y / c2)
        dt = (chi**3 * c3 + A * np.sqrt(y)) / np.sqrt(mu)

        # Derivative of dt with respect to z for Newton-Raphson
        if z == 0:
            dt_dz = (np.sqrt(2) / 40) * y**1.5 + (A / 8) * (np.sqrt(y) + A * np.sqrt(1 / (2 * y)))
        else:
            dt_dz = (1 / (2 * z)) * (dt - (3 * chi**3 * (c2 - (3 * c3) / (2 * c2)) + A * np.sqrt(y) * ((c2 - 2 * c3) / c2)) / np.sqrt(mu))


        # Newton-Raphson step
        z = z - (dt - tof) / dt_dz
        iterations += 1

    if iterations == max_iter:
        print("Warning: Maximum number of iterations reached.")
        print("Difference",abs(dt - tof))

    f = 1 - y / r1_norm
    g = A * np.sqrt(y / mu)

    g_dot = 1 - y / r2_norm
    f_dot = (f * g_dot - 1) / g

    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g

    return v1, v2

v1_sol, v2_sol = lambert_solver(r1, r2, tof, mu_sun)

print("Initial Velocity (v1):", v1_sol, "km/s")
print("Final Velocity (v2):", v2_sol, "km/s")

print("||v1|| =", np.linalg.norm(v1_sol), "km/s")
print("||v2|| =", np.linalg.norm(v2_sol), "km/s")
