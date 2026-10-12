from datetime import datetime, timedelta, timezone

import juliandate
import matplotlib.pyplot as plt
import numpy as np
import scipy
from data_classes import *
from orbit_data_query import query
from propagation import generate_ephemeris, generate_ephemeris_array, propagate_distance
from sgp4.api import SatrecArray
from sgp4.conveniences import jday


@dataclass
class CDM:
    # conjunction data message
    some_variable : str

def L_tca_sovler(julian_date_array : np.ndarray, primary_satrec: Satrec, secondary_satrec: Satrec, primary_space_object : SpaceObject, secondary_space_object : SpaceObject) -> float:
    """
        julian_date_array: (2*N_dates) -> (jd0,jfr0,jd1,jfr1,...)
    """
    jd = julian_date_array[0::2]
    jfr =julian_date_array[1::2]

    d = propagate_distance(jd, jfr, primary_satrec, secondary_satrec, primary_space_object, secondary_space_object) # (N_dates,)

    loss = np.sum(d) # minimize distance sum to minimize individual distances

    return loss

def conjunction_screening(primary_id : str, secondary_id : str, secondary_query_type : str = "NAME", screen_interval : int = 2) -> CDM:
    """
    screen for conjunctions for a primary satellite of interest over a time window of 2 days.

    primary_id : name of primary satelite for which conjunctions should be checked.
    secondary_id : ids or group of secondary satellites.
    secondary_query_type : query type
    screen_interval : scrren interval [hours]
    """

    # query primary
    primary_space_objects, primary_satrecs = query(primary_id, query_type="NAME") 
    primary_satrec = primary_satrecs[0]
    primary_space_object = primary_space_objects[0]

    # query secondary
    secondary_space_objects,secondary_satrecs = query(secondary_id, query_type=secondary_query_type)

    # remove primary from secondary if needed
    for so in secondary_space_objects:
        if so.norad_id == primary_space_object.norad_id:
            secondary_space_objects.remove(so)

    # propagate satrecs over 4h timewindow (ephimeris)
    start_time = datetime.datetime.now(timezone.utc)
    end_time = datetime.datetime.now(timezone.utc)+timedelta(days=screen_interval*4)
    dt = 2 # 2s interval [s]
    N = int((end_time - start_time) / timedelta(seconds=dt))
    rp, _, dp = generate_ephemeris(primary_satrec, primary_space_object, start_time, end_time, N) # (N_t [km], 3), (N_t [km/s], 3), (N_t [datetime])
    rs, _, _ = generate_ephemeris_array(SatrecArray(secondary_satrecs), secondary_space_objects, start_time, end_time, N) # (N_t [km], 3), (N_t [km/s], 3), (N_t [datetime])
    
    # calculate distances to primary object
    delta_r = np.broadcast_to(rp, (rs.shape[0], dp.shape[0], 3))-rs # (N_sat, N_t, 3) [km]
    #delta_v = np.broadcast_to(vp, (vs.shape[0], dp.shape[0], 3))-vs # (N_sat, N_t, 3) [km/s]

    # find minimas (close approaches)
    d = np.linalg.norm(delta_r, axis=2) # abs distance (N_sat,N_t) [km]
    d_dot = np.diff(d, axis=1) / dt
    mask = (d_dot[:, :-1] < 0) & (d_dot[:, 1:] > 0)
    minimas = [np.where(mask[i])[0] + 1 for i in range(d.shape[0])] # N_sat x (N_minima_i,)

    # use linear model for tca estimation
    tca_linears = []  # (N_sat,N_minima)
    d_tca_linears = [] # (N_sat,N_minima)
    for i, minima in enumerate(minimas):
        d1 = delta_r[i, minima, :]  # (N_minima, 3) [km]
        d2 = delta_r[i, minima+1, :] # (N_minima, 3) [km]
        delta_d = d2 - d1 # (N_minima, 3) [km]
        delta_tca_linear = - np.einsum('jk,jk->j', d1, delta_d) / np.linalg.norm(delta_d, axis=1)**2 * dt  # (N_minima,) [s]
        tca_linear = [dp[minima[i]]+timedelta(seconds=delta_tca_linear[i]) for i in range(len(minima))] # (N_minima,) [datetime]
        d_tca_linear = np.linalg.norm(d1 + delta_d * np.broadcast_to(delta_tca_linear[...,None], (delta_tca_linear.shape[0], 3))/dt, axis=1) # (N_minima,) [km]
        
        tca_linears.append(tca_linear)
        d_tca_linears.append(d_tca_linear)

    # use sgp4 optimization for tca estimation
    tca_solvers = [] # (N_sat, N_minima)
    d_tca_solvers = [] # (N_sat, N_minima)
    for secondary_satrec, secondary_space_object, minima in zip(secondary_satrecs, secondary_space_objects, minimas):
        minima_jds = [jday(*dp[m].timetuple()[:6]) for m in minima] # found minimas in julian format
        minima_jds = np.array([minima_jds[i][j] for i in range(len(minima_jds)) for j in [0,1]]) # flatten (2*N_minima,)

        lower_bound = [jday(*dp[minima-1][i].timetuple()[:6]) for i in range(len(minima))] # collect lower bounds
        lower_bound = np.array([lower_bound[i][j] for i in range(len(minima)) for j in [0,1]]) #flatten (2*N_minima,)
        upper_bound = [jday(*dp[minima+1][i].timetuple()[:6]) for i in range(len(minima))] # collect upper bounds
        upper_bound = np.array([upper_bound[i][j] for i in range(len(minima)) for j in [0,1]]) #flatten (2*N_minima,)
        bounds = [[lower_bound[i], upper_bound[i]] for i in range(len(minima_jds))]

        # find tca numerically
        loss_function = lambda julian_date_array: L_tca_sovler(julian_date_array, primary_satrec=primary_satrec, secondary_satrec=secondary_satrec, primary_space_object=primary_space_object, secondary_space_object=secondary_space_object)
        result = scipy.optimize.minimize(loss_function, minima_jds, bounds=bounds)
        tca_solver = []
        for jd, jfr in zip(result.x[0::2], result.x[1::2]):
            tca_solver.append(datetime.datetime(*juliandate.to_gregorian(jd+jfr), tzinfo=timezone.utc))        
        d_tca_solver = propagate_distance(result.x[0::2], result.x[1::2], primary_satrec, secondary_satrec, primary_space_object, secondary_space_object)

        tca_solvers.append(tca_solver)
        d_tca_solvers.append(d_tca_solver)


    for i in range(len(secondary_space_objects)):
        plt.scatter(dp, d[i])
        plt.scatter(tca_linears[i], d_tca_linears[i], label="linear_model")
        plt.scatter(tca_solvers[i], d_tca_solvers[i], label="numeric_solver")
        plt.legend()
        plt.show()
    

    # filter for close approaches (<100km)
    for i in range(len(secondary_space_objects)):
        d_ca = np.asarray(d_tca_solvers[i])[d_tca_solvers[i]<100]
        t_ca = np.asarray(tca_solvers[i])[d_tca_solvers[i]<100]
        print(d_ca)
        print(t_ca)

    # generate conjunction messages
    # alert for very close approaches
    # display closest apprach over all objects

    cdm = CDM("stuff")
    return cdm


if __name__ == "__main__":
    conjunction_screening("ISS (ZARYA)", "Tianhe")