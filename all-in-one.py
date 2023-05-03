import os

import pandas as pd
from matplotlib import pyplot as plt

from main import get_experiment_data_bars, get_experiment_metric

def get_concat_data(db):
    metr_concat = {}
    for node in db[0][1]:
        metr_concat[node] = pd.concat([db[0][1][node], pd.DataFrame({"timestampt": [None]}), db[1][1][node]],
                                      ignore_index=True)
    db_conc = (pd.concat([db[0][0], pd.DataFrame({"timestampt": [None]}), db[1][0]], ignore_index=True),
                metr_concat)
    return db_conc


def plot_all_in_one_conc1_2_4():
    db1 = get_experiment_data_bars('all-in-one/', 1, iteration=1, window=20)
    db2 = get_experiment_data_bars('all-in-one/', 2, iteration=4, window=20)
    db4 = get_experiment_data_bars('all-in-one/', 4, iteration=1, window=20)
    db8 = get_experiment_data_bars('all-in-one/', 8, iteration=4, window=20)
    db16 = get_experiment_data_bars('all-in-one/', 16, iteration=2, window=20)
    db64 = get_experiment_data_bars('all-in-one/', 64, iteration=1, window=20)
    dbs = [db1, db2, db4, db8, db16, db64]
    dbs_conc = list(map(lambda db: get_concat_data(db), dbs))


    concurrencies = [1, 2, 4, 8, 16, 64]

    plot_dur = True
    plot_mem = True

    if plot_dur:
        for i in range(6):
            # rally_data = pd.concat([dbs[i][0][0], pd.DataFrame({"timestampt":[None]}), dbs[i][1][0]], ignore_index=True)
            rally_data = dbs_conc[i][0]
            plt.plot(rally_data['hour'], rally_data["duration"], label=f"Concurrency {concurrencies[i]}", marker="o")
        ax = plt.gca()
        ax.set_ylim([-10, 550])
        ax.set_xlim([0, 39])
        plt.legend()
        plt.ylabel("Average successful workload duration (sec)")
        plt.xlabel("Experiment duration (hour)")
        fig = plt.gcf()
        fig.savefig('fig/AiO_duration.pdf')
        plt.show()

        for i in range(6):
            # rally_data = pd.concat([dbs[i][0][0], pd.DataFrame({"timestampt":[None]}), dbs[i][1][0]], ignore_index=True)
            rally_data = dbs_conc[i][0]
            plt.plot(rally_data['hour'], rally_data["successful_runs"], label=f"Concurrency {concurrencies[i]}", marker="o")
        plt.legend()
        plt.ylabel("Number of successful workload executions")
        plt.xlabel("Experiment duration (hour)")
        fig = plt.gcf()
        fig.savefig('fig/AiO_success.pdf')
        plt.show()

        for i in range(6):
            rally_data = dbs[i][0][0]
            rally_data_rej = dbs[i][1][0]
            initial_value = pd.concat([rally_data["duration"], rally_data_rej['duration']], ignore_index=True)[0]
            data = pd.concat([rally_data["duration"], rally_data_rej['duration']], ignore_index=True)
            plt.plot(((data - initial_value) / (initial_value / 100) + 100), label=f"Concurrency {concurrencies[i]}")
        plt.legend()
        plt.ylabel("Change of workload duration (%)")
        plt.title("Change of workload duration in relation to the beginning of the experiment")
        plt.show()

        for i in range(6):
            rally_data = dbs[i][0][0]
            rally_data_rej = dbs[i][1][0]
            initial_value = pd.concat([rally_data["successful_runs"], rally_data_rej['successful_runs']], ignore_index=True)[0]
            data = pd.concat([rally_data["successful_runs"], rally_data_rej['successful_runs']], ignore_index=True)
            plt.plot(((data - initial_value) / (initial_value / 100) + 100), label=f"Concurrency {concurrencies[i]}")
        plt.legend()
        plt.ylabel("Change of successful workloads (%)")
        plt.title("Change of successful workloads in relation to the beginning of the experiment")
        plt.show()

    if plot_mem:
        for i in range(6):
            metric_data = dbs_conc[i][1]
            control_node = list(metric_data.keys())[0]
            metric_data = metric_data[control_node]
            plt.plot(metric_data['hour'],
                     metric_data["node_memory_available_bytes_node_memory_MemAvailable_bytes"] / pow(2, 30),
                     label=f"Scenario {i + 7}", marker='o')
        plt.legend(loc="lower center", ncol=2)
        ax = plt.gca()
        ax.set_ylim([0, 4])
        plt.ylabel("Ram available (GB)")
        plt.xlabel("Experiment duration (hour)")
        fig = plt.gcf()
        fig.savefig('fig/AiO_memory_avail.pdf')
        plt.show()

        for i in range(6):
            metric_data = dbs_conc[i][1]
            control_node = list(metric_data.keys())[0]
            metric_data = metric_data[control_node]
            plt.plot(metric_data['hour'], metric_data["node_memory_swap_used_bytes"] / pow(2, 30), label=f"Scenario {i + 7}", linestyle="dashed", marker='o')
        plt.legend(loc="lower center")
        ax = plt.gca()
        ax.set_ylim([0, 5])
        plt.ylabel("Swaped used (GB)")
        plt.xlabel("Experiment duration (hour)")
        fig = plt.gcf()
        fig.savefig('fig/AiO_swap.pdf')
        plt.show()

    for i in range(6):
        metric_data = dbs_conc[i][1]
        control_node = list(metric_data.keys())[0]
        metric_data = metric_data[control_node]
        plt.plot(metric_data['hour'], metric_data["available_space_/dev/mapper/vg0-root_ext4_/"],
                 label=f"Exp. {i + 7}, swap used", linestyle="dashed", marker='o')
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([40, 110])
    plt.ylabel("Space left (%)")
    plt.show()




if not os.path.exists("fig"):
    os.makedirs("fig")
plt.rcParams['font.size'] = 16
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'][0] = 12

plot_all_in_one_conc1_2_4()
