from single_objective_optimization_03 import run_transfer
from numpy.linalg import norm

import pykep as pk
from pykep import lambert_problem, AU, DEG2RAD,DAY2SEC


departure = (2024, 12, 14, 0, 0, 0)
arrival   = (2025, 6, 8, 0, 0, 0)

t1 = pk.epoch_from_string(f"{departure[0]}-{departure[1]:02d}-{departure[2]:02d} {departure[3]:02d}:{departure[4]:02d}:{departure[5]:02d}")
t2 = pk.epoch_from_string(f"{arrival[0]}-{arrival[1]:02d}-{arrival[2]:02d} {arrival[3]:02d}:{arrival[4]:02d}:{arrival[5]:02d}")

# Time of flight in seconds
dt = (t2.mjd2000 - t1.mjd2000) * DAY2SEC

# Planet positions
earth = pk.planet.jpl_lp('earth')
rE, vE = earth.eph(t1)

mars = pk.planet.jpl_lp('mars')
rM, vM = mars.eph(t2)

# Solve Lambert problem
solver_pykep = pk.lambert_problem(r1=rE, r2=rM, tof=dt, mu=pk.MU_SUN, max_revs=1)
v_transfer_dep_pykep = solver_pykep.get_v1()#km/s
v_transfer_arr_pykep = solver_pykep.get_v2() #km/s
global_data=run_transfer(departure, arrival, bsp='de421.bsp', plot=False, summary=False)
v_transfer_dep = global_data['v_transfer_dep']
v_transfer_arr = global_data['v_transfer_arr']
print("########################### RESULTS USING PYKEP ######################## ")
print("||v_transfer_dep|| ={:.6f}".format(norm(v_transfer_dep_pykep)/ 1000.0 ))
print("||v_transfer_arr|| ={:.6f}".format(norm(v_transfer_arr_pykep)/ 1000.0  ))
print("########################### RESULTS USING THE LAMBERT SOLVER DEVELOPED ######################## ")

print("||v_transfer_dep|| ={:.6f}".format(norm(v_transfer_dep)))
print("||v_transfer_arr|| ={:.6f}".format(norm(v_transfer_arr)))