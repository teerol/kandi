import sys
import os
import json
import numpy as np

def calculate(json_f):
    # calculate performance measurements from trtexec timeslog json

    with open(json_f,'r') as f:
        s = f.read()
    d = json.loads(s)
    times = []
    print(f"{len(d)} lines in json")
    for i in d:
        times.append(i['endToEndMs'])

    times = np.array(times)
    avg = np.mean(times)
    mini = np.min(times)
    print(f"AVG time {avg:.2f}ms \nTHROURGHPUT: {1/avg*1000:.2f}FPS")
    print(f"MIN time {mini:.2f}ms \nTHROURGHPUT: {1/mini*1000:.2f}FPS")

if __name__ == '__main__':
    json_file = sys.argv[1]
    assert os.path.exists(json_file)
    calculate(json_file)