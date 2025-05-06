from tensorboard.backend.event_processing import event_accumulator
import os
import glob
import tqdm


def get_numbers(path: str):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()

    wall_times = []

    steps = 0

    # Iterate over all tags and collect wall_times
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        wall_times += [e.wall_time for e in events]
        steps = max(max([e.step for e in events]), steps)

    if len(wall_times) == 0:
        return 0,0
    uniq_times = sorted(list(set(wall_times)))
    start_time = uniq_times[0]
    total_time = 0

    last = start_time
    for i in uniq_times[1:]:
        diff_since_last = i - last
        if (diff_since_last < 300):
            total_time += diff_since_last
        last = i


    return total_time, steps

def process_all_event_files(directory):
    event_files = glob.glob(os.path.join(directory, "**", "events.out.tfevents.*"), recursive=True)
    results = {}
    for file in tqdm.tqdm(event_files):
        results[file] = get_numbers(file)
    return results

data = process_all_event_files(r"rl\logs")
with open(r"thesis_images\times.csv", "w") as f:
    for file, d in data.items():
        time, steps = d
        f.write(f"{file}\t{time}\t{steps}\n")

#get_numbers(r"C:\Users\Jindra\Documents\GitHub\WebElementDetector\wed\rl\logs\v4ResNetFrozen\PPO_1\events.out.tfevents.1733436536.DESKTOP-80I2VIK.25052.0")