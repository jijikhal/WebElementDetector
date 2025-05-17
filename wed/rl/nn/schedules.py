import math

def exp_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return math.exp(-2*(1-progress_remaining))*initial_value*0.6

    return func

## Taken from rl-zoo
def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func

def custom_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        total_timesteps = 10_000_000  # Example: Total training steps
        current_step = int((1 - progress_remaining) * total_timesteps)
        if current_step < 500_000:
            return initial_value
        elif current_step < 1_150_000:
            return initial_value * (1 - (current_step-500_000) / 700_000)
        elif current_step < 1_600_000:
            return initial_value * 0.1 * (1 - (current_step-1_150_000) / 500_000)
        else:
            return initial_value * 0.01

    return func