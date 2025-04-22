"""Pipeline frequency optimizer implementation.

The `PipelineFrequencyOptimizer` is to be integrated into the training framework.
It is responsible for communicating with the PFO server and managing
the `FrequencyController` instance, which is responsible for controlling
the frequency of the CPU of the current process.
"""

from __future__ import annotations
from typing import Optional
import multiprocessing as mp
from multiprocessing import Event
from pynvml import nvmlInit

import httpx
import torch
import torch.distributed as dist
import time
import queue

from zeus.callback import Callback
from zeus.device import get_gpus
from zeus.optimizer.pipeline_frequency.frequency_controller import FrequencyController
from zeus.optimizer.pipeline_frequency.common import (
    GET_FREQUENCY_SCHEDULE_URL,
    REGISTER_JOB_URL,
    REGISTER_RANK_URL,
    REPORT_TIMING_URL,
    REPORT_ENERGY_URL,
    REPORT_PROFILING_RESULT_URL,
    REPORT_SCHEDULE_URL,
    JobInfo,
    RankInfo,
    FrequencySchedule,
)
from zeus.utils.framework import sync_execution

# To be Removed
import logging

logger = logging.getLogger(__name__)

def generate_pipe_schedule(num_microbatches: int) -> list[str]:
            """Generate a pipeline schedule list.
            
            For each microbatch, we expect a forward followed by a backward instruction.
            For example, if num_microbatches is 10, this returns a list with 20 items:
            ["forward", "backward", "forward", "backward", ..., "forward", "backward"]
            """
            return ["forward", "backward"] * num_microbatches


class PipelineFrequencyOptimizer(Callback):
    """Pipeline frequency optimizer."""

    def __init__(
        self,
        rank: int,
        dp_rank: int,
        pp_rank: int,
        tp_rank: int,
        device_id: int,
        dp_degree: int,
        pp_degree: int,
        tp_degree: int,
        world_size: int,
        server_url: str,
        job_metadata: str | None = None,
        partition_method: Optional[str] = None,
        microbatch_size: Optional[int] = None,
        num_microbatches: Optional[int] = None,
        num_prof_steps: Optional[int] = None,
        warmup_iters: Optional[int] = None,
    ) -> None:
        """Initialize the Pipeline frequency optimizer.

        Assumptions:
            - `torch.distributed` has been initialized.
            - `torch.cuda.set_device` has been called with `device_id`.
                This is needed to broadcast the job ID to all ranks.

        The master process (rank 0) will register the job with the Peresus
        server and retrieve the job ID of this job. Then, each rank will
        report itself to the PFO server with the job ID.

        Args:
            rank: Global rank of the current process.
            dp_rank: Rank in the data parallel group.
            pp_rank: Rank in the pipeline parallel group.
            tp_rank: Rank in the tensor parallel group.
            device_id: CUDA device ID that the current process manages.
            dp_degree: Size of the data parallel group.
            pp_degree: Size of the pipeline parallel group.
            tp_degree: Size of the tensor parallel group.
            world_size: Total number of ranks that participate in training.
            server_url: URL of the PFO server.
            job_metadata: An optional arbitrary string that describes the job. This will
                be appended to the job ID if given. Typically for logging purposes.
        """
        if not dist.is_initialized():
            raise RuntimeError(
                "Instantiate `PipelineFrequencyOptimizer` after `init_process_group`."
            )

        self.server_url = server_url
        self.rank = rank
        self.dp_rank = dp_rank
        self.pp_rank = pp_rank
        self.tp_rank = tp_rank
        self.device_id = device_id

        # pick defaults if user didn’t specify
        DEFAULT_MICROBATCH_SIZE = 12
        DEFAULT_NUM_MICROBATCHES = 4
        DEFAULT_NUM_PROF_STEPS = 4
        DEFAULT_WARMUP_ITERS = 2

        self.microbatch_size = microbatch_size  or DEFAULT_MICROBATCH_SIZE
        self.num_microbatches = num_microbatches or DEFAULT_NUM_MICROBATCHES
        self.num_prof_steps = num_prof_steps     or DEFAULT_NUM_PROF_STEPS
        self.warmup_iters = warmup_iters         or DEFAULT_WARMUP_ITERS

        gpus = get_gpus()
        torch.cuda.set_device(device_id)

        # Rank 0 registers the job with the PFO server and retrieves the job ID.
        job_id = None
        if rank == 0:
            job_info = JobInfo(
                pp_degree=pp_degree,
                dp_degree=dp_degree,
                tp_degree=tp_degree,
                world_size=world_size,
                job_metadata=job_metadata,
                framework="PyTorch",
                model_name="TestModel",
                partition_method=partition_method,
                microbatch_size=self.microbatch_size,
                num_microbatches=self.num_microbatches,
                num_prof_steps=self.num_prof_steps,
                warmup_iters=self.warmup_iters,
            )

            # print("\nSending job_info:", job_info.dict())
            response = httpx.post(self.server_url + REGISTER_JOB_URL, json=job_info.dict())
            # print("\nRequest headers:", response.request.headers)
            if (code := response.status_code) != 200:
                raise RuntimeError(
                    f"PFO server returned status code {code}: {response.text}"
                )
            job_id = response.json()
            if not isinstance(job_id, str):
                raise RuntimeError(f"PFO server returned a strange job ID: {job_id=}")

        # Rank 0 broadcasts the job ID across all ranks.
        objects = [job_id]
        dist.broadcast_object_list(objects, src=0)
        self.job_id = objects[0]
        if self.job_id is None:
            raise RuntimeError("Failed to broadcast job ID to all ranks")

        # Query the list of available frequencies of the GPU.
        max_mem_freq = max(gpus.getSupportedMemoryClocks(device_id))
        freqs = sorted(
            gpus.getSupportedGraphicsClocks(device_id, max_mem_freq),
            reverse=True,
        )

        # Query the pipeline schedule.
        num_microbatches = num_microbatches if num_microbatches is not None else DEFAULT_NUM_MICROBATCHES
        pipe_schedule = generate_pipe_schedule(num_microbatches)

        # Each rank reports itself to the PFO server with the job ID.
        rank_info = RankInfo(
            rank=self.rank,
            dp_rank=self.dp_rank,
            pp_rank=self.pp_rank,
            tp_rank=self.tp_rank,
            available_frequencies=freqs,
            pipe_schedule=pipe_schedule,
        )
        response = httpx.post(
            self.server_url + REGISTER_RANK_URL.format(job_id=self.job_id),
            json=rank_info.dict(),
        )
        if (code := response.status_code) != 200:
            raise RuntimeError(
                f"PFO server returned status code {code}: {response.text}"
            )

        # The frequency controller is responsible for controlling the frequency
        # of the GPU (device_id) asynchronously.
        self.frequency_controller = FrequencyController(device_id=device_id)

        # Fetch the frequency schedule from the PFO server.
        self.freq_schedule = self._get_frequency_schedule()
        self.freq_schedule_iter = iter(self.freq_schedule)

        # Containers for timing and energy data.
        self.timing_data = {"forward": [], "backward": []}
        self.energy_data = []

        # print("Energy polling to be called now", flush=True)
        # Start the energy polling process.
        ctx = mp.get_context('spawn')
        self._stop_event = ctx.Event()
        self._energy_queue = ctx.Queue()

        self.energy_polling_process = ctx.Process(
            target=_energy_polling_loop,
            args=(device_id, self._stop_event, self._energy_queue),
            daemon=True
        )
        self.energy_polling_process.start()

    def _get_frequency_schedule(self) -> list[tuple[str, int]]:
        """Get the frequency schedule from the PFO server."""
        # print("\n Fetching frequency schedule from PFO server... \n", flush=True)
        logger.info("Fetching frequency schedule from PFO server.")
        response = httpx.get(
            self.server_url + GET_FREQUENCY_SCHEDULE_URL.format(job_id=self.job_id),
            params={"rank": self.rank},
            timeout=None,
        )
        if (code := response.status_code) != 200:
            raise RuntimeError(
                f"PFO server returned status code {code}: {response.text}"
            )
        schedule = FrequencySchedule.parse_raw(response.text)
        if schedule.rank != self.rank:
            raise RuntimeError(
                f"PFO server returned a schedule for rank {schedule.rank} to rank {self.rank}"
            )
        logger.info("\n\n\nFrequency schedule: %d", len(schedule.frequencies))
        # print("Frequency schedule length:", len(schedule.frequencies), schedule, flush=True)
        
        return schedule.frequencies
    
    

    def on_step_begin(self) -> None:
        """Mark the beginning of a step."""
        self._step_start_time = time.time()

    def on_step_end(self) -> None:
        """Mark the end of a step.

        Also report the profiling result to the PFO server after N iterations.
        """
        self.freq_schedule = self._get_frequency_schedule()
        self.freq_schedule_iter = iter(self.freq_schedule)
        sched_payload = FrequencySchedule(
            rank=self.rank,
            frequencies=self.freq_schedule
        )
        httpx.post(
            f"{self.server_url}{REPORT_SCHEDULE_URL.format(job_id=self.job_id)}",
            json=sched_payload.dict(),
            timeout=5
        )

    def on_instruction_begin(self, name: str) -> None:
        """Mark the beginning of an instruction, like forward and backward.

        Retrieve the next frequency from the schedule, check whether the next
        expected instruction matches the name of the instruction, and set the
        frequency accordingly.
        """
        sync_execution([self.device_id], sync_with="torch")
        self._instr_start_time = time.time()
        # Retrieve the next frequency from the schedule.
        item = next(self.freq_schedule_iter, None)
        if item is None:
            raise RuntimeError("PFO server returned fewer frequencies than expected")

        # Check whether the next expected instruction matches the name of the instruction.
        instruction, frequency = item
        if instruction != name:
            logging.warning(
                f"The next expected instruction does not match the passed name: {name}, instruction: {instruction}"
            )
        try:
            self.frequency_controller.set_frequency(frequency)
        except Exception as e:
            logging.warning(f"PFO: failed to set GPU freq {frequency}: {e}")

    def on_instruction_end(self, name: str) -> None:
        """Mark the end of an instruction, like forward and backward and report its latency."""
        end_time = time.time()
        self.timing_data.setdefault(name, []).append((self._instr_start_time, end_time))

    def on_train_end(self) -> None:
        """Clean up when training is complete and send aggregated energy data."""
        self.collect_energy()
        self.stop_energy_polling()
        # Create and send timing data payload to the server in one HTTP request.
        timing_payload = {
            "job_id": self.job_id,
            "rank": self.rank,
            "timing_breakdown": self.timing_data,
        }
        try:
            timing_url = (
            f"{self.server_url}"
            f"{REPORT_TIMING_URL.format(job_id=self.job_id)}"
            )
            httpx.post(timing_url, json=timing_payload, timeout=5)
            
            logger.info("Timing data successfully sent.")
        except Exception as e:
            logger.error("Error sending timing data:", e)

        # Send aggregated energy data payload to the server in one HTTP request.
        energy_payload = {
            "job_id": self.job_id,
            "rank": self.rank,
            "energy_measurements": self.energy_data,
        }
        try:
            energy_url = (
            f"{self.server_url}"
            f"{REPORT_ENERGY_URL.format(job_id=self.job_id)}"
            )
            httpx.post(energy_url, json=energy_payload, timeout=5)
           
            logger.info("Energy data successfully sent.")
        except Exception as e:
            logger.error("Error sending energy data:", e)

    def collect_energy(self):
        """Drain any pending measurements into your in‑memory list."""
        while True:
            try:
                tag, payload = self._energy_queue.get_nowait()
            except queue.Empty:
                # No more items → clean exit
                break
            except Exception as e:  # queue empty
                logger.info("Energy polling error: %s", e)
                break

            if tag == "data":
                timestamp, joules = payload
                # print("\nCollecting energy into energy_data\n", flush=True)
                self.energy_data.append((timestamp, joules))
            else:
                logger.error("Energy polling error: %s", payload)

    def stop_energy_polling(self) -> None:
        """Stop the energy polling process gracefully."""
        if hasattr(self, "energy_polling_process"):
            self._stop_event.set()
            self.energy_polling_process.join()
            


def _energy_polling_loop(device_id: int,
                         stop_event: mp.Event,
                         out_queue: mp.Queue,
                        ):
        """Continuously poll energy and aggregate measurements for later reporting."""
        try:
            nvmlInit()
        except Exception as e:
            logger.error("Failed to init NVML in polling process: %s", e)
            return
        
        gpus = get_gpus()
        prev_measurement = None
        while True:
            # Poll continuously
            curr_time = time.time()
            curr_energy = gpus.getTotalEnergyConsumption(device_id) / 1000.0
            measurement = (curr_time, curr_energy)
            # Deduplicate: only add if measurement is different from the last one.
            if measurement != prev_measurement:
                # sending measurement in ctx.queue
                out_queue.put(("data", measurement))
                prev_measurement = measurement
            if stop_event.is_set():
                # print("\nStop Event Detected, stopping energy polling\n", flush=True)
                break