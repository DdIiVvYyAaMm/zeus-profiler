"""Controller that sets the GPU's frequency in a non-blocking fashion."""

from __future__ import annotations

import atexit
import contextlib
import multiprocessing as mp
from multiprocessing import get_context
from pynvml import nvmlInit

from zeus.device import get_gpus
from zeus.device.gpu import ZeusGPUNotSupportedError


class FrequencyController:
    """Spawns a separate process that sets the GPU frequency."""

    def __init__(self, device_id: int = 0) -> None:
        """Instantiate the frequency controller.

        Args:
            device_id: Device ID of the GPU to control.
        """
        ctx = get_context('spawn')  
        self._q: mp.Queue[int | None] = ctx.Queue()
        self._proc = ctx.Process(
            target=self._controller_process,
            args=(device_id,),
            daemon=True
        )
        atexit.register(self.end)
        self._proc.start()

    def set_frequency(self, frequency: int) -> None:
        """Set the GPU's frequency asynchronously.

        If `frequency` is zero, returns without doing anything.
        """
        if frequency > 0:
            try:
                self._q.put(frequency, block=False)
            except Exception:
                pass

    def end(self) -> None:
        """Stop the controller process."""
        try:
            self._q.put(None, block=False)
        except Exception:
            pass


    def _controller_process(self, device_id: int) -> None:
        """Receive frequency values through a queue and apply it."""
        try:
            gpus = get_gpus()

            # Reset any custom power‐limit if supported
            with contextlib.suppress(Exception):
                gpus.resetPowerManagementLimit(device_id)

            # Lock memory clock to max if supported
            max_mem = None
            with contextlib.suppress(Exception):
                mems = gpus.getSupportedMemoryClocks(device_id)
                if mems:
                    max_mem = max(mems)
                    gpus.setMemoryLockedClocks(device_id, max_mem, max_mem)

            # Lock GPU (SM) clock to max if supported
            current = None
            with contextlib.suppress(Exception):
                if max_mem is not None:
                    gfxs = gpus.getSupportedGraphicsClocks(device_id, max_mem)
                else:
                    gfxs = gpus.getSupportedGraphicsClocks(device_id)
                if gfxs:
                    top = max(gfxs)
                    gpus.setGpuLockedClocks(device_id, top, top)
                    current = top

            # Loop: wait for new freq or shutdown
            while True:
                target = self._q.get(block=True)
                if target is None:
                    break
                if current != target:
                    with contextlib.suppress(Exception):
                        gpus.setGpuLockedClocks(device_id, target, target)
                        current = target

        finally:
            # On exit, reset any locks
            with contextlib.suppress(Exception):
                gpus.resetMemoryLockedClocks(device_id)
            with contextlib.suppress(Exception):
                gpus.resetGpuLockedClocks(device_id)