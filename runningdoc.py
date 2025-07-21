import subprocess
import re

NUM_RUNS = 50  # Set how many times you want to run the simulation
win_count = 0

for i in range(NUM_RUNS):
    print(f"Running simulation {i+1}/{NUM_RUNS}...")
    proc = subprocess.run(
        [
            "uv", "run", "python3", "match_simulator.py",
            "--submissions", "1:example_submissions/.py", "3:wena10_phase31.py", "--engine"
        ],
        capture_output=True, text=True
    )
    output = proc.stdout + proc.stderr

    # Parse the ranking line
    match = re.search(r"outcome was \{.*?ranking=\[([0-9, ]+)\]", output)
    if match:
        ranking = [int(x.strip()) for x in match.group(1).split(",")]
        if ranking[0] == 0:
            win_count += 1

print(f"\nPlayer 0 winrate: {win_count}/{NUM_RUNS} = {win_count/NUM_RUNS:.2%}")