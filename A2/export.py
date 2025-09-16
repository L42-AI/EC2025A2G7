import numpy as np

from ctm_types import History

def export_results(file_path: str, history: History) -> None:
    """Export the simulation results to a CSV file.

    Parameters
    ----------
    file_path : str
        The path to the output CSV file.
    history : list
        The history of positions or other relevant data to be exported.

    Returns
    -------
    None
        This function writes the history data to a CSV file.
    """
    np.savetxt(file_path, history, delimiter=",")