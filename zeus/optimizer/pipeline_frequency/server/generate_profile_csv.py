"""Generate a CSV profile that combines timing and energy data."""

from collections import defaultdict
import os
import csv
import numpy as np


class PiecewiseLinearModel:
    """A simple model that interpolates energy measurements over time."""

    def __init__(
        self, time_measurements: np.ndarray, energy_measurements: np.ndarray
    ) -> None:
        """Initilaize a piecewise linear model that interpolates energy measurements over time.

        Args:
        time_measurements: 1D array of timestamps.
        energy_measurements: 1D array of energy readings corresponding to the timestamps.
        """
        self.times = time_measurements
        self.energies = energy_measurements
        # Ensure measurements are sorted.
        if not np.all(np.diff(self.times) >= 0):
            raise ValueError("Time measurements must be sorted in ascending order.")
        if not np.all(np.diff(self.energies) >= 0):
            raise ValueError("Energy measurements must be sorted in ascending order.")

    def __call__(self, t: float) -> float:
        """Return the interpolated energy reading at time t.

        Raises ValueError if t is out of the measurement range.
        """
        if t < self.times[0] or t > self.times[-1]:
            raise ValueError(
                f"Time {t} is out of range [{self.times[0]}, {self.times[-1]}]."
            )
        return np.interp(t, self.times, self.energies).item()

def generate_profile_csv(
    job_id: str,
    timing_data: dict[int, dict[str, list[tuple[float, float]]]],
    energy_data: dict[int, list[tuple[float, float]]],
    dump_dir: str,
    num_microbatches: int,
    num_prof_steps: int,
    warmup_iters: int,
    frequency_schedule: dict[int, list[int]] = None,
) -> str:
    """
    Generate a CSV profile that combines timing and energy measurements.

    For each rank and for each instruction type (e.g. 'forward', 'backward'),
    this function skips an initial number of warmup iterations (warmup_iters × batch_size)
    and then aggregates measurements in batches (batch_size = num_microbatches × num_prof_steps).
    The energy consumption during an instruction is estimated by constructing a piecewise
    linear model from the energy measurements and computing the difference between the energy
    readings at the end and at the start of the instruction.

    Optionally, if a frequency schedule is provided (as a list of frequency values per rank),
    the function will record the frequency applied for each batch.

    Args:
        job_id: The unique job identifier.
        timing_data: Dictionary mapping rank to a dict of instruction names to lists of
                     (start_time, end_time) tuples.
        energy_data: Dictionary mapping rank to a list of (time, energy) measurement tuples.
        dump_dir: Directory where the CSV file will be saved.
        num_microbatches: Number of microbatches per iteration.
        num_prof_steps: Number of profiling steps per iteration.
        warmup_iters: Number of warmup iterations to skip.
        frequency_schedule: Dictionary mapping rank to a list of frequency values for each batch.

    Returns:
        The file path of the generated CSV.
    """
    os.makedirs(dump_dir, exist_ok=True)
    output_path = os.path.join(dump_dir, f"{job_id}_profile.csv")

    # Build interpolation models per rank
    models = {}
    for rank, measurements in energy_data.items():
        arr = np.array(measurements)
        models[rank] = PiecewiseLinearModel(arr[:,0], arr[:,1])

    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["rank", "instruction", "frequency", "avg_time", "avg_energy"])

        # For each rank, flatten and sort the timing events
        for rank, inst_map in timing_data.items():
            model = models.get(rank)
            if model is None:
                continue

            # Build chronological list of valid events
            events: list[tuple[float,str,float,float]] = []
            for inst, spans in inst_map.items():
                for start, end in spans:
                    try:
                        de = model(end) - model(start)
                    except ValueError:
                        # skip instructions outside energy-sample range
                        continue
                    dt = end - start
                    events.append((start, inst, dt, de))

            # sort by start time
            events.sort(key=lambda e: e[0])

            batch_size = num_microbatches * num_prof_steps
            skip = warmup_iters * batch_size
            if skip < len(events):
                events = events[skip:]
            else:
                events = []

            # Grab the schedule frequencies for this rank
            freqs = frequency_schedule.get(rank, []) if frequency_schedule else []
            if len(freqs) < len(events):
                freqs = freqs + ["N/A"] * (len(events) - len(freqs))

            # Zip measurements and schedule
            assigned = [
                (inst, freq, dt, de)
                for (_, inst, dt, de), freq in zip(events, freqs)
            ]

            # Group by (rank, inst, freq) and average
            grouped: dict[tuple[int,str,int], list[tuple[float,float]]] = defaultdict(list)
            for inst, freq, dt, de in assigned:
                key = (rank, inst, freq if isinstance(freq, int) else -1)
                grouped[key].append((dt, de))

            # Write one row per group, sorted by descending frequency
            for (r, inst, freq), vals in sorted(
                grouped.items(),
                key=lambda item: (item[0][0], item[0][1], -item[0][2])
            ):  
                times, energies = zip(*vals)
                if freq == "N/A" or freq<0 or float(np.mean(energies))==0.0:
                    continue
                writer.writerow([
                    r,
                    inst,
                    freq,
                    float(np.mean(times)),
                    float(np.mean(energies)),
                ])

    return output_path
