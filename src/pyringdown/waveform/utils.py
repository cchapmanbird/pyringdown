from typing import Optional

def get_sampling_parameters_from_partial_inputs(Nt: Optional[int]=None, dt: Optional[float]=None, T: Optional[float]=None):
    T_temp = T
    dt_temp = dt
    Nt_temp = Nt

    if (Nt is None) and (dt is None) and (T is None):
        raise ValueError("At least two of Nt, dt, or T must be provided.")

    elif (Nt is not None) and (dt is not None):
        T_temp = Nt * dt
        T = T_temp if T is None else T

    elif (Nt is not None) and (T is not None):
        dt_temp = T / Nt
        dt = dt_temp if dt is None else dt

    elif (dt is not None) and (T is not None):
        Nt_temp = int(T / dt)
        Nt = Nt_temp if Nt is None else Nt
        
    if (Nt != Nt_temp) or (dt != dt_temp) or (T != T_temp):
        raise ValueError(
            f"""
            Provided sampling parameters Nt={Nt}, dt={dt}, T={T} are inconsistent with
            Nt={Nt_temp}, dt={dt_temp}, T={T_temp}.
            """
        )

    return Nt_temp, dt_temp, T_temp