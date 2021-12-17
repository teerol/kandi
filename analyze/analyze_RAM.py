import matplotlib.pyplot as plt
import sys
import os

def import_ram_data(file):
    with open(file, 'r') as f:
        d = []
        for line in f:
            d.append(int(line.split(' ')[1].split('/')[0]))
    return d


def main(filename):
    # Analyze tegrastats logfile
    # makes plot of used RAM and prints MIN and MAX values 

    tegra_interval = 5 #seconds
    data = import_ram_data(filename)
    print(f"MAX: {max(data)}MB, MIN {min(data)}MB")

    x = range(0, len(data)*tegra_interval, tegra_interval)
    plt.plot(x, data)
    plt.plot([min(x),max(x)], [max(data),max(data)],"k--")
    plt.xlabel("aika (s)")
    plt.ylabel("RAM-muistia käytössä (Mb)")
    name = os.path.basename(filename).split(".")[0]
    plt.savefig(f"mem_{name}.png")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("ERROR, GIVE FILENAME TO ANALYZE")
    else:
        file = sys.argv[1]
        assert os.path.exists(file)
        main(file)