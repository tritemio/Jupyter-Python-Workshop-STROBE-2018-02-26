import numpy as np
import pandas as pd


def compute_states_startstop(blink_trajectory):
    """Compute start-stop times of residency in a state.

    This function takes as input an array of booleans `blink_trajectory`
    and computes the start-stop times for all the time ranges where the state
    is "on", i.e. where `blink_trajectory` is True.

    Implementation:
        This version goes through `blink_trajectory` array element 
        by element using a for-loop which is slow.

    Arguments:
        blink_trajectory (array of bool): "blinking trajectory", telling
            whether the state is populated (True) or not (False) as a
            function of time.

    Return:
        A list of 3-element tuples (start, stop, label), one for each set of
        consecutive time-points in the trajectory where the state is "on".
    """
    start_stop_list = []
    prev_state_on = False
    for i, curr_state_on in enumerate(blink_trajectory):
        if curr_state_on and not prev_state_on:
            start = i
        elif not curr_state_on and prev_state_on:
            stop = i
            start_stop_list.append((start, stop))
        prev_state_on = curr_state_on
    df = pd.DataFrame(start_stop_list, columns=['istart', 'istop'])
    return df


def compute_states_startstop_numpy(blink_trajectory):
    """Compute start-stop times of residency in a state.

    This function takes as input an array of booleans `blink_trajectory`
    and computes the start-stop times for all the time ranges where the state
    is "on", i.e. where `blink_trajectory` is True.

   Implementation:
        This version uses numpy's vectorial operations to avoid a for-loop
        and is faster than the python version.

    Arguments:
        blink_trajectory (array of bool): "blinking trajectory", telling
            whether the state is populated (True) or not (False) as a
            function of time.

    Return:
        A list of 3-element tuples (start, stop, label), one for each set of
        consecutive time-points in the trajectory where the state is "on".
    """
    d = np.diff(blink_trajectory.view('int8'))
    on_start = np.where(d > 0)[0] + 1
    on_stop = np.where(d < 0)[0]
    if on_start[0] > on_stop[0]:
        on_stop = on_stop[1:]
    if on_start[-1] > on_stop[-1]:
        on_start = on_start[:-1]
    on_start_stop = np.vstack((on_start, on_stop))
    df = pd.DataFrame(on_start_stop.T, columns=['istart', 'istop'])
    return df
