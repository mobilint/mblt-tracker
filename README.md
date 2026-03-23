# Mobilint Device Tracker

<!-- markdownlint-disable MD033 -->
<div align="center">
<p>
<a href="https://www.mobilint.com/" target="_blank">
<img src="https://raw.githubusercontent.com/mobilint/mblt-tracker/master/assets/Mobilint_Logo_Primary.png" alt="Mobilint Logo" width="60%">
</a>
</p>
</div>
<!-- markdownlint-enable MD033 -->

**mblt-tracker** is a Python package to track device metrics, such as [Mobilint NPU](https://www.mobilint.com/aries), NVIDIA GPU, Intel CPU, ...

Designed to help developers measure the device performance with fair criteria.

## Installation

[![PyPI - Version](https://img.shields.io/pypi/v/mblt-tracker?logo=pypi&logoColor=white)](https://pypi.org/project/mblt-tracker/)
[![PyPI Downloads](https://static.pepy.tech/badge/mblt-tracker?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://clickpy.clickhouse.com/dashboard/mblt-tracker)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mblt-tracker?logo=python&logoColor=gold)](https://pypi.org/project/mblt-tracker/)

- Install **mblt-tracker** using pip:

```bash
pip install mblt-tracker
```

- If you want to install the latest version from the source, clone the repository and install it:

```bash
git clone https://github.com/mobilint/mblt-tracker.git
cd mblt-tracker
pip install -e .
```

## Quick Start Guide

This package is designed to use as the following workflow.

```python
from mblt_tracker import NPUDeviceTracker # or CPUDeviceTracker or GPUDeviceTracker


your_tracker = NPUDeviceTracker(interval=0.1)
# Do some other stuff
# such as setting, warming up, etc.

your_tracker.start() # Start tracking device metrics
# Run inference code that you want to measure device usage while running

your_tracker.stop() # Stop tracking device metrics

# Get the metric
metric = your_tracker.get_metric()
print(metric)

# Get the trace
trace = your_tracker.get_trace()
print(trace)

# Reset the tracker
your_tracker.reset() # Reset internal sampled data
```

## Mobilint NPU

> Note: Prepare environment equipped with Mobilint NPU. In case you are not a Mobilint customer, please contact [us](mailto:tech-support@mobilint.com).

The prior function of this tracker is measuring Mobilint NPU's power, memory usage and utilization.

Since we are not currently supporting Python interface to NPU management and monitoring functions, this tracker is based on 

## Intel CPU

Our CPU tracker is based on ['PyRAPL'](https://pypi.org/project/pyRAPL/) and [`psutil`](https://pypi.org/project/psutil/).

If your enviroment equipped with multiple CPUs, 

### Troubleshooting

```bash
sudo chmod -R a+r /sys/class/powercap/intel-rapl/

--privileged
```

## NVIDIA GPU

Our GPU tracker is based on [`pyNVML`](https://pypi.org/project/nvidia-ml-py/).

If your enviroment equipped with multiple GPUs, 