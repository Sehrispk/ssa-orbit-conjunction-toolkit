import numpy as np
from data_classes import SpaceObject
from sgp4.api import SGP4_ERRORS, SatrecArray, jday
from sgp4.conveniences import dump_satrec
from sys import stdout
from datetime import timezone, timedelta, datetime

def propagate_satrecs(satrecs : SatrecArray, julian_dates : np.ndarray, julian_date_fractions: np.ndarray, space_objects : list[SpaceObject]) -> tuple[np.ndarray, np.ndarray]:
    julian_dates = np.array(julian_dates)
    julian_date_fractions = np.array(julian_date_fractions)
    if len(julian_dates.shape) != 1 and len(julian_date_fractions.shape) != 1:
        raise ValueError("Invalid date format. Dates should be 1-dimensional arrays.")

    e, r, v = satrecs.sgp4(julian_dates, julian_date_fractions)
    if e.any():
        error_codes = e[e!=0]
        faulty_objects = np.array(space_objects)
        faulty_objects = faulty_objects[e[:,0]!=0]
        print(f"Error codes detected: {e[:,0]}")
        for code, object in zip(error_codes, faulty_objects):
            print(SGP4_ERRORS[code] + f" for space object {object.name} ({object.intdes}, {object.norad_id})")
            stdout.writelines(dump_satrec(object.satrec))

    return r, v

def generate_ephemeris(satrecs, space_objects, start_date: datetime, end_date:datetime, N):
    dates = [start_date + (end_date - start_date) * t for t in np.linspace(0,1,N)]

    jds = []
    jfrs = []
    for date in dates:
        jd, jfr = jday(*date.timetuple()[:6])
        jds.append(jd)
        jfrs.append(jfr)

    r, v = propagate_satrecs(satrecs, np.array(jds), np.array(jfrs), space_objects)

    return r, v, dates

if __name__ == "__main__":
    from orbit_data_query import query

    space_objects, satrecs = query("STATIONS", query_type="GROUP")

    jd, jfr = jday(2026, 10, 10, 12, 0, 0)

    r, v = propagate_satrecs(satrecs, np.array([jd,]), np.array([jfr,]), space_objects)


    start_time = datetime.now(timezone.utc)
    end_time = datetime.now(timezone.utc)+timedelta(days=2)
    dt = 2 # 2s interval
    N = int((end_time - start_time) / timedelta(seconds=2))

    r, v, d = generate_ephemeris(satrecs, space_objects, start_time, end_time, N)
