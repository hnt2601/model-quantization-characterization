import pynvml
import statistics


class PowerConsumption():
    def __init__(self, device_id=0):
        pynvml.nvmlInit()
        # device_count = pynvml.nvmlDeviceGetCount()
        self.device_id = device_id
        self.power_usages = []
        self.energy_usages = []

    def __del__(self):
        pynvml.nvmlShutdown()

    def measure_power_usage(self, duration):
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
        power_usage = (
            pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        )  # Convert to watts

        self.power_usages.append(power_usage)

        power_usage = power_usage * duration
        self.energy_usages.append(power_usage)

    def get_consumption(self):
        power_consumption = statistics.mean(self.power_usages)
        energy_consumption = statistics.mean(self.energy_usages)

        return power_consumption, energy_consumption
