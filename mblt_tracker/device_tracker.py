from abc import ABC, abstractmethod
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_RUNNING


class BaseDeviceTracker(ABC):
    """Abstract base class for device trackers."""

    def __init__(self, interval: float, **kwargs):
        """Initialize the device tracker.

        Args:
            interval (float): The interval in seconds at which the device should be polled.
            **kwargs: Additional keyword arguments for parent classes.

        Raises:
            ValueError: If the interval is not positive.
        """
        super().__init__(**kwargs)
        if interval <= 0:
            raise ValueError("interval must be positive")
        self._interval = interval
        self._scheduler: Optional[BackgroundScheduler] = None
        self._job_id: str = "device_track"

    def start(self) -> None:
        """Start tracking device metrics periodically."""
        self.reset()
        try:
            self._func_for_sched()
        except Exception:
            pass
        if self._scheduler is None or self._scheduler.state != STATE_RUNNING:
            self._scheduler = BackgroundScheduler()
            self._scheduler.start()

        # Ensure _scheduler is not None for the linter
        if self._scheduler is not None:
            if self._scheduler.get_job(self._job_id) is not None:
                self._scheduler.remove_job(self._job_id)
            self._scheduler.add_job(
                self._func_for_sched,
                "interval",
                seconds=self._interval,
                id=self._job_id,
            )

    def stop(self) -> None:
        """Stop tracking device metrics and shut down the scheduler."""
        try:
            self._func_for_sched()
        except Exception:
            pass
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=True)
            finally:
                self._scheduler = None

    @abstractmethod
    def _func_for_sched(self) -> None:
        """Scheduled function to capture a single sample of metrics."""

    @abstractmethod
    def get_metric(self) -> dict:
        """Return summarized device metrics since start or last reset.

        Returns:
            Dict: A dictionary containing averaged and peak metrics.
        """

    @abstractmethod
    def get_trace(self) -> list[tuple[float, float]]:
        """Return a time-series trace of power measurements.

        Returns:
            List[Tuple[float, float]]: A list of (timestamp, power_watt) pairs.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all collected metrics and traces."""
