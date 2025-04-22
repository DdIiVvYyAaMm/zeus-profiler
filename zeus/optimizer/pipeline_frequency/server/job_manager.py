"""The JobManager singleton class manages all job states."""

from __future__ import annotations

import time
import asyncio
import traceback

from fastapi import HTTPException

from zeus.optimizer.pipeline_frequency.common import (
    JobInfo,
    PFOServerSettings,
    FrequencySchedule,
    ProfilingResult,
    RankInfo,
    save_prof,
    save_sched,
    save_ranks,
)

# from zeus.optimizer.pipeline_frequency.server.router import (
#     TimingBreakdownData,
#     EnergyMeasurementData,
# )
from zeus.optimizer.pipeline_frequency.server.generate_profile_csv import (
    generate_profile_csv,
)

from zeus.utils.logging import get_logger
from zeus.utils.async_utils import create_task

GLOBAL_JOB_MANAGER: JobManager | None = None

logger = get_logger(__name__)

logger.info(">>> LOADED JobManager from %s", __file__)

class JobManager:
    """A singleton class that manages all states."""

    def __init__(self, pfo_settings: PFOServerSettings) -> None:
        """Initialize the job manager."""
        self.pfo_settings = pfo_settings

        self._job_infos: dict[str, JobInfo] = {}
        self._job_rank_infos: dict[str, list[RankInfo]] = {}
        self._job_tasks: dict[str, asyncio.Task] = {}
        # self._job_result_channels: dict[str, asyncio.Queue[ProfilingResult]] = {}
        # self._job_sched_request_channels: dict[str, asyncio.Queue] = {}
        # self._job_sched_response_channels: dict[str, list[asyncio.Queue]] = {}
        self._job_last_active_time: dict[str, float] = {}
        self._job_timing_data: dict[
            str, dict[int, dict[str, list[tuple[float, float]]]]
        ] = {}
        self._job_energy_data: dict[str, dict[int, list[tuple[float, float]]]] = {}
        self._job_frequency_data: dict[str, dict[int, list[int]]] = {}

        # Spawn cleanup task that evicts the state of jobs that have not been active
        # for a long time.
        create_task(
            self._cleanup_task(
                cleanup_period=60,
                max_idle_time=pfo_settings.max_job_idle_time,
            ),
            logger=logger,
        )

    def register_job(self, job_info: JobInfo) -> None:
        """Prepare internal state for a new job.

        This method will be invoked exactly once by the global rank 0 (master) process.
        """
        job_id = job_info.job_id
        world_size = job_info.world_size
        self._job_infos[job_id] = job_info
        self._job_rank_infos[job_id] = []
        # self._job_result_channels[job_id] = asyncio.Queue(maxsize=world_size)
        # self._job_sched_request_channels[job_id] = asyncio.Queue(maxsize=world_size)
        # self._job_sched_response_channels[job_id] = [
        #     asyncio.Queue(maxsize=1) for _ in range(world_size)
        # ]
        self._job_tasks[job_id] = create_task(
            self._job_task(job_id, self.pfo_settings.dump_data),
            logger=logger,
        )
        self._job_last_active_time[job_id] = time.monotonic()
        self._job_timing_data[job_id] = {}
        self._job_energy_data[job_id] = {}
        self._job_frequency_data[job_id] = {}

    def register_rank(self, job_id: str, rank_info: RankInfo) -> None:
        """Register rank-specific information for an already registered job.

        This method will be invoked `world_size` number of times (once per rank).
        """
        self._job_rank_infos[job_id].append(rank_info)
        self._job_last_active_time[job_id] = time.monotonic()

    async def get_frequency_schedule(
        self, job_id: str, rank: int
    ) -> FrequencySchedule:
        """
        Return *every* frequency this rank supports, so the client can
        locally profile them all.
        """
        # 1) sanity
        if job_id not in self._job_rank_infos:
            raise HTTPException(404, f"Unknown job {job_id}")
        # 2) pick out the RankInfo
        infos = self._job_rank_infos[job_id]
        ri = next((r for r in infos if r.rank == rank), None)
        if ri is None:
            raise HTTPException(404, f"Rank {rank} not registered in job {job_id}")
        # 3) build a big schedule: for *each* freq, run through the pipe_schedule
        schedule: list[tuple[str,int]] = []
        for freq in ri.available_frequencies:
            schedule.extend((inst, freq) for inst in ri.pipe_schedule)

        return FrequencySchedule(rank=rank, frequencies=schedule)

    def report_profiling_result(self, job_id: str, result: ProfilingResult) -> None:
        """Send the profiling result to the job task and immediately return.

        This method will be called `world_size` number of times - one for each rank.
        """
        self._job_result_channels[job_id].put_nowait(result)
        self._job_last_active_time[job_id] = time.monotonic()

    def report_timing(self, data) -> None:
        """Receive timing breakdown data from the client.

        `data` should contain fields: job_id, rank, timing_breakdown.
        """
        job_id = data.job_id
        rank = data.rank
        self._job_timing_data.setdefault(job_id, {})[rank] = data.timing_breakdown
        self._job_last_active_time[job_id] = time.monotonic()
        logger.info("Timing breakdown reported for job %s, rank %d", job_id, rank)

    def report_energy(self, data) -> None:
        """Receive energy measurement data from the client.

        `data` should contain fields: job_id, rank, energy_measurements.
        """
        job_id = data.job_id
        rank = data.rank
        self._job_energy_data.setdefault(job_id, {})[rank] = data.energy_measurements
        self._job_last_active_time[job_id] = time.monotonic()
        logger.info("Energy data reported for job %s, rank %d", job_id, rank)

    def report_schedule(self, job_id: str, schedule: FrequencySchedule) -> None:
        rank = schedule.rank
        # Extract only frequencies
        self._job_frequency_data[job_id][rank] = [freq for _, freq in schedule.frequencies]
        # self._job_last_active_time[job_id] = time.monotonic()
        logger.info("Frequency schedule reported for job %s", rank)
        
    async def _cleanup_task(
        self,
        cleanup_period: int,
        max_idle_time: int,
    ) -> None:
        while True:
            await asyncio.sleep(cleanup_period)
            now = time.monotonic()
            for job_id, last in list(self._job_last_active_time.items()):
                if now - last > max_idle_time:
                    # Evict all state
                    for d in (
                        self._job_infos,
                        self._job_rank_infos,
                        self._job_timing_data,
                        self._job_energy_data,
                        self._job_frequency_data,
                        self._job_last_active_time,
                        # self._job_sched_request_channels,
                        # self._job_sched_response_channels,
                    ):
                        d.pop(job_id, None)
                        
    async def _job_task(self, job_id: str, dump_data: bool) -> None:
        """
        Waits until all ranks have reported RankInfo, timing, energy, and frequency,
        then generates the profile CSV and exits.
        """
        job_info = self._job_infos[job_id]
        world_size = job_info.world_size
        dump_dir = f"{self.pfo_settings.dump_dir}/{job_id}"

        try:
            # 1) Wait for all ranks to register
            while len(self._job_rank_infos[job_id]) < world_size:
                await asyncio.sleep(0.1)

            # 2) Optionally save rank metadata
            if dump_data:
                await save_ranks(self._job_rank_infos[job_id], dump_dir)

            # 3) Wait for timing, energy, and frequency data
            while True:
                if (
                    len(self._job_timing_data.get(job_id, {})) == world_size
                    and len(self._job_energy_data.get(job_id, {})) == world_size
                    and len(self._job_frequency_data.get(job_id, {})) == world_size
                ):
                    break
                await asyncio.sleep(0.1)

            # 4) Generate CSV profile
            num_microbatches = job_info.num_microbatches
            num_prof_steps = job_info.num_prof_steps
            warmup_iters = job_info.warmup_iters

            timing = self._job_timing_data[job_id]
            energy = self._job_energy_data[job_id]
            freq_map = self._job_frequency_data[job_id]

            logger.info(f"Calling Profile CSV generation.")
            
            csv_path = generate_profile_csv(
                job_id,
                timing,
                energy,
                dump_dir,
                num_microbatches,
                num_prof_steps,
                warmup_iters,
                frequency_schedule=freq_map,
            )
            logger.info(f"Profile CSV generated at: {csv_path}")

        except asyncio.CancelledError:
            # This task gets cancelled when it's idle for too long and evicted.
            pass

        except Exception as e:
            logger.error(f"Error in job_task for job {job_id}: {e}")


def init_global_job_manager(pfo_settings: PFOServerSettings) -> None:
    """Instantiate the global singleton `JobManager`."""
    global GLOBAL_JOB_MANAGER
    GLOBAL_JOB_MANAGER = JobManager(pfo_settings=pfo_settings)


def get_global_job_manager() -> JobManager:
    """Fetch the global singleton `JobManager`."""
    assert GLOBAL_JOB_MANAGER is not None, "`init_global_job_manager` was not called."
    return GLOBAL_JOB_MANAGER
