"""Shared constants and models between the server and the client (optimizer)."""

from __future__ import annotations

import os
import inspect
from datetime import datetime
from typing import Any, Optional, List

import aiofiles
import pandas as pd

from zeus.utils.pydantic_v1 import BaseModel, BaseSettings, Field, validator, PyObject

from enum import Enum

GET_SERVER_INFO_URL = "/info"
REGISTER_JOB_URL = "/register_job"
REGISTER_RANK_URL = "/register_rank/{job_id}"
GET_FREQUENCY_SCHEDULE_URL = "/schedule/{job_id}"
REPORT_PROFILING_RESULT_URL = "/result/{job_id}"


class PFOServerSettings(BaseSettings):
    """PFO server settings, configurable via environment variables.

    For instance, setting `ZEUS_PFO_LOG_LEVEL=INFO` will automatically set
    the `log_level` variable to `"INFO"`.

    Attributes:
        scheduler: Name of the `FrequencyScheduler` to use.
        scheduler_args: Any extra arguments required by `scheduler.__init__`.
        log_level: Log level, e.g. "debug", "info".
        dump_data: Whether the scheduler should dump internal state to the filesystem
            (for future inspection purposes).
        dump_dir: Directory to dump state in (if enabled)
        max_job_idle_time: Maximum time in seconds that a job can be idle for before
            its states are automatically deleted from the server.
    """

    scheduler: PyObject = "PointSolution"  # type: ignore
    scheduler_args: dict[str, Any] = {}
    log_level: str = "DEBUG"
    dump_data: bool = True
    dump_dir: str = "./dump"
    max_job_idle_time: int = 60 * 60 * 24 * 7  # 1 week

    @validator("scheduler", pre=True)
    def _fix_scheduler_import_path(cls, value):
        """Prepend `zeus.optimizer.pipeline_frequency.server.scheduler.` to the scheduler type name."""
        return f"zeus.optimizer.pipeline_frequency.server.scheduler.{value}"

    @validator("scheduler_args")
    def _validate_scheduler_args(cls, args, values):
        """Check whether args are as expected by the scheduler's constructor."""
        scheduler = values["scheduler"]
        full_args = args | dict(job_info=None, rank_infos=None, pfo_settings=None)
        constructor_args = inspect.signature(scheduler)
        try:
            constructor_args.bind(**full_args)
        except TypeError as e:
            raise ValueError(f"Invalid scheduler args: {e}") from None
        return args

    @validator("log_level")
    def _make_upper_case(cls, value):
        return value.upper()

    class Config:  # type: ignore
        """Configuration class read by pydantic."""

        env_prefix = "zeus_pfo_"


class JobInfo(BaseModel):
    """Training job information reported to the server.

    Attributes:
        job_id: Globally unique ID of the training job, generated by the server.
            This field should be an empty string when sent to the server.
        pp_degree: Pipeline parallel degree.
        dp_degree: Data parallel degree.
        tp_degree: Tensor parallel degree.
        world_size: World size of the training job.
        job_metadata: An optional arbitrary string that describes the job. This will
            be appended to the job ID if given. Typically for logging purposes.
    """

    job_id: str = ""
    pp_degree: int = Field(ge=1)
    dp_degree: int = Field(ge=1)
    tp_degree: int = Field(ge=1)
    world_size: int = Field(ge=1)
    job_metadata: Optional[str] = None

    # New fields needed by InstructionProfiler
    framework: Optional[str] = ""
    model_name: Optional[str] = ""
    partition_method: Optional[str] = ""
    microbatch_size: Optional[int] = None
    num_microbatches: Optional[int] = None

    @validator("job_id")
    def _check_empty_job_id(cls, job_id):
        assert not job_id
        return job_id

    @validator("world_size")
    def _check_world_size(cls, world_size, values):
        """Product of PP, DP, and TP degree would be identical to the world size."""
        assert (
            values["pp_degree"] * values["dp_degree"] * values["tp_degree"]
            == world_size
        )
        return world_size

    def set_job_id(self, scheduler_name: str):
        """Generate and set the job ID."""
        self.job_id = "+".join(
            [
                datetime.now().strftime("%F-%H-%M-%S"),
                self.framework or "",
                self.model_name or "",
                self.partition_method or "",
                f"dp{self.dp_degree}",
                f"pp{self.pp_degree}",
                f"tp{self.tp_degree}",
                f"mbs{self.microbatch_size!s}",
                f"nmb{self.num_microbatches!s}",
                scheduler_name or "",
            ]
        )
        if self.job_metadata:
            self.job_id += f"+{self.job_metadata}"


class RankInfo(BaseModel):
    """Information passed to the server from each rank.

    Attributes:
        rank: Global rank of the reporting process.
        dp_rank: Data parallel rank of the reporting procees.
        pp_rank: Pipeline parallel rank of the reporting procees.
        tp_rank: Tensor parallel rank of the reporting procees.
        available_frequencies: List of available frequencies for the rank's GPU.
    """

    rank: int = Field(ge=0)
    dp_rank: int = Field(ge=0)
    pp_rank: int = Field(ge=0)
    tp_rank: int = Field(ge=0)
    available_frequencies: list[int]

    # New fields for InstructionProfiler:
    pipe_schedule: List[PipeInstruction] = []  # to be filled by PipeInstruction
    power_state_range: List[int] = []  # list of power states to try

    @validator("power_state_range")
    def _validate_power_state_and_sort(cls, value):
        if value is not None:
            if any(ps <= 0 for ps in value):
                raise ValueError("Power state values must be positive integers.")
            if len(value) != len(set(value)):
                raise ValueError("List of power states must be unique.")
            return sorted(value, reverse=True)
        else:
            return value


# New for InstructionProfiler
class PipeInstruction(str, Enum):
    """Atomic operations in pipeline schedules."""

    LOAD = "load"
    FORWARD = "forward"
    BACKWARD = "backward"
    P2P = "p2p"
    CC = "cc"
    STEP = "step"
    LOW = "low"
    HIGH = "high"
    OTHER = "other"
    PURE_BACKWARD = "pure_backward"


# New for InstructionProfiler
class PowerStateSchedule(BaseModel):
    """Power state assignment for each PipeInstruction of a rank."""

    rank: int = Field(ge=0)
    power_states: List[int]


class FrequencySchedule(BaseModel):
    """Frequency schedule for one iteration.

    `frequencies` is a list of tuples, where the first element is the name of the
    instruction and the second element is the frequency to use for that instruction.
    """

    rank: int = Field(ge=0)
    frequencies: list[tuple[str, int]]


class ProfilingResult(BaseModel):
    """Profiling results for a `FrequencySchedule` of a rank.

    Attributes:
        rank: Global rank of the reporting client.
        iter_time: List of latency of all iterations within the profiling window in seconds.
        iter_energy: List of energy consumption of all iterations within the profiling window in Joules.
        time_breakdown: Duration of each operation across multiple iterations.
            e.g. `time_breakdown["forward"][i]` is the list of latencies of all forward computations
            in the `i`th iteration.
        energy_breakdown: Energy consumption of each operation across multple iterations.
            Value has the same structure as `time_breakdown`.
    """

    rank: int = Field(ge=0)
    iter_time: list[float]
    iter_energy: list[float]
    time_breakdown: dict[str, list[list[float]]] = {}
    energy_breakdown: dict[str, list[list[float]]] = {}


class OfflineProfilingResult(BaseModel):
    """Profiling results generated from offline profiling each instruction.

    Attributes:
        rank: Global rank of the reporting client.
        dp_rank: Data parallel rank of the reporting procees.
        pp_rank: Pipeline parallel rank of the reporting procees.
        tp_rank: Tensor parallel rank of the reporting procees.
        forward_time: Dict that maps frequency to average forward computation time.
        forward_energy: Dict that maps frequency to average forward computation energy.
        backward_time: Dict that maps frequency to average backward computation time.
        backward_energy: Dict that maps frequency to average backward computation energy.
    """

    rank: int = Field(ge=0)
    dp_rank: int = Field(ge=0)
    pp_rank: int = Field(ge=0)
    tp_rank: int = Field(ge=0)
    forward_time: dict[int, float]
    forward_energy: dict[int, float]
    backward_time: dict[int, float]
    backward_energy: dict[int, float]


class InstructionProfilingResult(BaseModel):
    """Time and energy profiling results for each instruction in each stage."""

    __root__: list[OfflineProfilingResult]

    def to_csv(self, filepath: str) -> None:
        """Serialize and save this object into a CSV file.

        Columns: rank, dp_rank, pp_rank, tp_rank, stage, instruction, frequency, time, energy
        Notes
            - `rank` is the global rank of the process.
            - `pp_rank` and `stage` are always the same, for backwards compatibility.
            - All ranks and `stage` are zero-indexed.
            - `instruction` is either "forward" or "backward".
            - `time` and `energy` are already averaged over profiling iterations.
        """
        if not filepath.endswith(".csv"):
            raise ValueError("Filepath does not end with '.csv'")

        # fmt: off
        headers = ["rank", "dp_rank", "pp_rank", "tp_rank", "stage", "instruction", "frequency", "time", "energy"]
        records: list[tuple[int, int, int, int, int, str, int, float, float]] = []
        for res in self.__root__:
            prefix = (res.rank, res.dp_rank, res.pp_rank, res.tp_rank, res.pp_rank)
            for freq in res.forward_time:
                records.append((*prefix, "forward", freq, res.forward_time[freq], res.forward_energy[freq]))
            for freq in res.backward_time:
                records.append((*prefix, "backward", freq, res.backward_time[freq], res.backward_energy[freq]))
        # fmt: on

        df = pd.DataFrame.from_records(records, columns=headers)
        df.to_csv(filepath, index=False)


async def save_prof(
    data: list[ProfilingResult],
    directory: str,
    schedule_num: int,
) -> None:
    """Save a list of `ProfilingResult`s in the designated directory."""
    os.makedirs(directory, exist_ok=True)
    async with aiofiles.open(f"{directory}/{schedule_num}.prof.json", "w") as f:
        obj = _ProfilingResultList(__root__=data).json()
        await f.write(obj)


def load_prof(directory: str, schedule_num: int) -> list[ProfilingResult]:
    """Load a list of `ProfilingResult`s saved in the designated directory."""
    filepath = f"{directory}/{schedule_num}.prof.json"
    return _ProfilingResultList.parse_file(filepath).__root__


async def save_sched(
    data: list[FrequencySchedule],
    directory: str,
    schedule_num: int,
) -> None:
    """Save a list of `FrequencySchedule`s in the designated directory."""
    os.makedirs(directory, exist_ok=True)
    async with aiofiles.open(f"{directory}/{schedule_num}.sched.json", "w") as f:
        obj = _FrequencyScheduleList(__root__=data).json()
        await f.write(obj)


def load_sched(directory: str, schedule_num: int) -> list[FrequencySchedule]:
    """Load a list of `FrequencySchedule`s saved in the designated directory."""
    filepath = f"{directory}/{schedule_num}.sched.json"
    return _FrequencyScheduleList.parse_file(filepath).__root__


async def save_ranks(data: list[RankInfo], directory: str) -> None:
    """Save a list of `RankInfo`s in the designated directory."""
    os.makedirs(directory, exist_ok=True)
    async with aiofiles.open(f"{directory}/ranks.json", "w") as f:
        obj = _RankInfoList(__root__=data).json()
        await f.write(obj)


def load_ranks(directory: str) -> list[RankInfo]:
    """Load a list of `RankInfo`s saved in the designated directory."""
    filepath = f"{directory}/ranks.json"
    return _RankInfoList.parse_file(filepath).__root__


# Proxy classes for a list of Pydantic objects.
# __root__ is making use of Pydantic's Custom Root Type for a cleaner JSON representation.


class _ProfilingResultList(BaseModel):
    __root__: list[ProfilingResult]


class _FrequencyScheduleList(BaseModel):
    __root__: list[FrequencySchedule]


class _RankInfoList(BaseModel):
    __root__: list[RankInfo]
