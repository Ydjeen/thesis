import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from main import get_experiment_metric, get_experiment_data_bars, get_bar_change, extract_rally_output

scenario_word = "Scenario"
scenario_duration = "Scenario duration (hour)"
connect_rejuv=True

def plot_HA_conc1_2_4():
    db1 = get_experiment_data_bars('high-availability/', 1, iteration=1, window=20)
    db2 = get_experiment_data_bars('high-availability/', 2, iteration=1, window=20)
    db4 = get_experiment_data_bars('high-availability/', 4, iteration=1, window=20)

    metr_concat = {}
    for node in db1[0][1]:
        metr_concat[node] = pd.concat([db1[0][1][node], pd.DataFrame({"timestampt": [None]}), db1[1][1][node]], ignore_index=True)
    db1_conc = (pd.concat([db1[0][0], pd.DataFrame({"timestampt":[None]}), db1[1][0]], ignore_index=True),
                metr_concat)
    metr_concat = {}
    for node in db2[0][1]:
        metr_concat[node] = pd.concat([db2[0][1][node], pd.DataFrame({"timestampt": [None]}), db2[1][1][node]], ignore_index=True)
    db2_conc = (pd.concat([db2[0][0], pd.DataFrame({"timestampt":[None]}), db2[1][0]], ignore_index=True),
                metr_concat)
    metr_concat = {}
    for node in db4[0][1]:
        metr_concat[node] = pd.concat([db4[0][1][node], pd.DataFrame({"timestampt": [None]}), db4[1][1][node]], ignore_index=True)
    db4_conc = (pd.concat([db4[0][0], pd.DataFrame({"timestampt":[None]}), db4[1][0]], ignore_index=True),
                metr_concat)

    m1 = get_experiment_metric('high-availability/', 1, iteration=1, window=20)
    m2 = get_experiment_metric('high-availability/', 2, iteration=1, window=20)
    m4 = get_experiment_metric('high-availability/', 4, iteration=1, window=20)

    dbs = [db1, db2, db4]
    dbs_conc = [db1_conc, db2_conc, db4_conc]
    concurrencies= [1,2,4]
    for i in range(3):
        #rally_data = pd.concat([dbs[i][0][0], pd.DataFrame({"timestampt":[None]}), dbs[i][1][0]], ignore_index=True)
        rally_data = dbs_conc[i][0]
        rally_data['hour'][rally_data['hour'] > 24] = 26
        if connect_rejuv:
            rally_data=rally_data.drop(24)
        plt.plot(rally_data['hour'], rally_data["duration"], label=f"Scenario {i+1}", marker="o")
    plt.grid()
    plt.legend()
    plt.ylabel("Average workload duration (sec)")
    plt.xlabel("Scenario duration (hour)")
    plt.grid()
    fig = plt.gcf()
    fig.savefig('fig/HA_1_2_4_duration.pdf')
    plt.show()

    for i in range(3):
        metric_data = dbs_conc[i][1]
        control_node = list(metric_data.keys())[0]
        metric_data = metric_data[control_node]
        metric_data['hour'][metric_data['hour'] > 24] = 26
        if connect_rejuv:
            metric_data=metric_data.drop(24)
        plt.plot(metric_data['hour'], metric_data["node_memory_available_bytes_node_memory_MemAvailable_bytes"]/pow(2,30), label=f"Scenario {i+1}, RAM available", color=f"C{i}", marker='o')
        plt.plot(metric_data['hour'], metric_data["node_memory_swap_used_bytes"]/pow(2,30), label=f"Scenario {i+1}, swap used", linestyle="dashed", color=f"C{i}", marker='v')
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(0.0, 0.32))
    ax = plt.gca()
    ax.set_ylim([0, 4.5])
    plt.ylabel("Average memory usage (GB)")
    plt.xlabel("Scenario duration (hour)")
    plt.grid()
    fig = plt.gcf()
    fig.savefig('fig/HA_1_2_4_memory.pdf')
    plt.show()

    return

    for i in range(3):
        rally_data = dbs[i][0][0]
        rally_data['hour'][rally_data['hour'] > 24] = 26
        plt.plot(rally_data['hour'], rally_data["duration"].pct_change().cumsum(), label=f"Scenario {i}")
    plt.grid()
    plt.legend()
    plt.ylabel("Change of workload duration (%)")
    plt.title("Change of workload duration in relation to the beginning of the scenario")
    plt.grid()
    plt.show()

    for i in range(3):
        metric_data = dbs[i][0][1]
        control_node = list(metric_data.keys())[0]
        metric_data = metric_data[control_node]
        metric_data['hour'][metric_data['hour'] > 24] = 26
        plt.plot(metric_data['hour'], metric_data["node_memory_available_bytes_node_memory_MemAvailable_bytes"]/pow(2,30), label=f"Scenario {i}")
    plt.grid()
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([0, 6])
    plt.ylabel("Control node, RAM available (GB)")
    plt.grid()
    plt.show()

    for i in range(3):
        metric_data = dbs[i][0][1]
        control_node = list(metric_data.keys())[0]
        metric_data = metric_data[control_node]
        plt.plot(metric_data['hour'], metric_data["node_memory_swap_used_bytes"]/pow(2,30), label=f"Concurrency {concurrencies[i]}")
    plt.grid()
    plt.legend()
    plt.ylabel("Control node swap used (GB)")
    plt.grid()
    plt.show()

    for i in range(3):
        rally_data = dbs[i][0][0]
        plt.plot(rally_data['hour'], rally_data["successful_runs"], label=f"Scenario {concurrencies[i]}")
    plt.grid()
    plt.legend()
    plt.ylabel("Amount of successfull workloads/h")
    plt.grid()
    plt.show()

def plot_HA_conc8():
    #failure timestamp = 1681179738
    m = get_experiment_metric('high-availability/', 8, iteration=1, window=20)

    control_df = m[list(m.keys())[0]]
    plt.figure().set_figwidth(12)
    plt.plot(control_df['hour'], control_df["node_memory_available_bytes_node_memory_MemAvailable_bytes"] / pow(2, 30),
             label="RAM available")
    plt.plot(control_df['hour'], control_df["node_memory_swap_used_bytes"] / pow(2, 30), label="Swap used")
    plt.axvline(x=18.5, color='r', label='OpenStack fails')
    plt.grid()
    plt.xlabel("Scenario duration (hour)")
    plt.ylabel("Memory usage (GB)")
    plt.legend()
    fig = plt.gcf()
    fig.savefig('fig/HA_8_memory.pdf')
    plt.show()

    nodes = ["Control node", "Monitoring node", "Compute node 1", "Compute node 2"]
    plt.figure().set_figwidth(12)
    for i, node in enumerate(m):
        plt.plot(m[node]['hour'], m[node]["available_space_/dev/mapper/vg0-root_ext4_/"], label=nodes[i])
    plt.axvline(x=18.5, color='r', label='OpenStack fails')
    plt.grid()

    plt.xlabel("Scenario duration (hour)")
    plt.ylabel("Free physical space left (%)")
    plt.legend()
    fig = plt.gcf()
    fig.savefig('fig/HA_8_space.pdf')
    plt.show()

    s = get_experiment_data_bars('high-availability/', 8, iteration=1, window=20)
    s[0] = (s[0][0], {node: frame for node, frame in s[0][1].items()})
    control_df = m[list(m.keys())[0]]
    plt.figure().set_figwidth(12)
    plt.bar(s[0][0]['hour'], s[0][0]["duration"])
    plt.axvline(x=19, color='r', label='OpenStack fails')
    plt.grid()
    plt.xlabel("Scenario duration (hour)")
    plt.ylabel("Average workload duration (sec)")
    plt.legend()
    fig = plt.gcf()
    fig.savefig('fig/HA_8_duration.pdf')
    plt.show()

def plot_HA_conc16():
    db1 = get_experiment_data_bars('high-availability/', 16, iteration=1, window=20)

    metr_concat = {}
    for node in db1[0][1]:
        metr_concat[node] = pd.concat([db1[0][1][node], pd.DataFrame({"timestampt": [None]}), db1[1][1][node]], ignore_index=True)
    db1_conc = (pd.concat([db1[0][0], pd.DataFrame({"timestampt":[None]}), db1[1][0]], ignore_index=True),
                metr_concat)
    metr_concat = {}

    m1 = get_experiment_metric('high-availability/', 1, iteration=1, window=20)

    dbs = [db1]
    dbs_conc = [db1_conc]
    concurrencies= [16]
    for i in range(1):
        #rally_data = pd.concat([dbs[i][0][0], pd.DataFrame({"timestampt":[None]}), dbs[i][1][0]], ignore_index=True)
        rally_data = dbs_conc[i][0]
        plt.plot(rally_data['hour'], rally_data["duration"], label=f"Scenario {i}", marker="o")
    plt.grid()
    plt.legend()
    plt.ylabel("Average workload duration (sec)")
    plt.xlabel("Scenario duration (hour)")
    plt.grid()
    plt.show()

    for i in range(1):
        rally_data = dbs[i][0][0]
        plt.plot(rally_data['hour'], rally_data["duration"].pct_change().cumsum(), label=f"Scenario {i}")
    plt.grid()
    plt.legend()
    plt.ylabel("Change of workload duration (%)")
    plt.title("Change of workload duration in relation to the beginning of the scenario")
    plt.grid()
    plt.show()

    for i in range(1):
        metric_data = dbs[i][0][1]
        control_node = list(metric_data.keys())[0]
        metric_data = metric_data[control_node]
        plt.plot(metric_data['hour'], metric_data["node_memory_available_bytes_node_memory_MemAvailable_bytes"]/pow(2,30), label=f"Scenario {i}")
    plt.grid()
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([0, 6])
    plt.ylabel("Control node, RAM available (GB)")
    plt.grid()
    plt.show()

    for i in range(1):
        metric_data = dbs_conc[i][1]
        control_node = list(metric_data.keys())[0]
        metric_data = metric_data[control_node]
        plt.plot(metric_data['hour'], metric_data["node_memory_available_bytes_node_memory_MemAvailable_bytes"]/pow(2,30), label=f"Concurrency {concurrencies[i]}, RAM available", color=f"C{i}", marker='o')
        plt.plot(metric_data['hour'], metric_data["node_memory_swap_used_bytes"]/pow(2,30), label=f"Concurrency {concurrencies[i]}, swap used", linestyle="dashed", color=f"C{i}", marker='v')
    plt.grid()
    plt.legend()
    ax = plt.gca()
    ax.set_ylim([0, 4.5])
    plt.ylabel("Memory usage (GB)")
    plt.xlabel(scenario_duration)
    plt.grid()
    plt.show()

    for i in range(1):
        metric_data = dbs[i][0][1]
        control_node = list(metric_data.keys())[0]
        metric_data = metric_data[control_node]
        plt.plot(metric_data['hour'], metric_data["node_memory_swap_used_bytes"]/pow(2,30), label=f"Concurrency {concurrencies[i]}")
    plt.grid()
    plt.legend()
    plt.ylabel("Control node swap used (GB)")
    plt.grid()
    plt.show()

    for i in range(1):
        rally_data = dbs_conc[0][0]
        plt.plot(rally_data['hour'], rally_data["successful_runs"], label=f"Scenario {concurrencies[i]}", marker="o")
    plt.grid()
    plt.legend()
    plt.ylabel("Amount of successfull workloads/h")
    plt.grid()
    plt.show()

    rally_data = extract_rally_output('high-availability/', 16, iteration=1)
    first_execution_df = pd.DataFrame(rally_data[0])
    second_execution_df = pd.DataFrame(rally_data[1])
    first_execution_df.columns = ['timestamp', 'duration', 'error', 'actions']
    second_execution_df.columns = ['timestamp', 'duration', 'error', 'actions']
    fe_timestamp_start = first_execution_df['timestamp'].min()
    fe_timestamp_end = first_execution_df['timestamp'].max()
    first_hour_chunk = first_execution_df[first_execution_df['timestamp'].between(fe_timestamp_start, fe_timestamp_start+3600)]
    first_hour_chunk['timestamp'] = first_hour_chunk['timestamp'] - fe_timestamp_start
    windows_size = int((len(first_hour_chunk)/60))
    data = list()
    for i in range(60):
        chunk = first_hour_chunk[first_hour_chunk['timestamp'].between((60*i), (60*(i+1)))]
        data.append({"minute":i+0.5 ,"timestamp":chunk['timestamp'].mean(),
                                     "successful_runs": chunk['error'].where(chunk['error'].apply(lambda x: len(x)) == 0, other=np.nan).count(),
                                     "failed_runs": chunk['error'].where(chunk['error'].apply(lambda x: len(x)) > 0, other=np.nan).count()})
    data = pd.DataFrame(data)
    #success = first_hour_chunk['duration'].where(first_hour_chunk['error'].apply(lambda x: len(x)) == 0, other=np.nan).rolling(windows_size, center = True).count()
    #errors = first_hour_chunk['duration'].where(first_hour_chunk['error'].apply(lambda x: len(x)) > 0, other=np.nan).rolling(windows_size, center = True).count()
    plt.stackplot(data['minute'],
                  data['successful_runs'],
                  data["failed_runs"],
                  labels=["Successfull workloads", "Failed workloads"])
    plt.ylabel("Amount of workloads executed")
    plt.xlabel("Scenario duration (min)")
    plt.legend()
    plt.grid()
    fig = plt.gcf()
    fig.savefig('fig/HA_16_first_hour.pdf')
    plt.show()

    se_timestamp_start = second_execution_df['timestamp'].min()
    first_hour_chunk = second_execution_df[
        second_execution_df['timestamp'].between(se_timestamp_start, se_timestamp_start + 3600)]
    first_hour_chunk['timestamp'] = first_hour_chunk['timestamp'] - se_timestamp_start
    data = list()
    for i in range(60):
        chunk = first_hour_chunk[first_hour_chunk['timestamp'].between((60*i), (60*(i+1)))]
        data.append({"minute":i+0.5 ,"timestamp":chunk['timestamp'].mean(),
                                     "successful_runs": chunk['error'].where(chunk['error'].apply(lambda x: len(x)) == 0, other=np.nan).count(),
                                     "failed_runs": chunk['error'].where(chunk['error'].apply(lambda x: len(x)) > 0, other=np.nan).count()})
    data = pd.DataFrame(data)
    #success = first_hour_chunk['duration'].where(first_hour_chunk['error'].apply(lambda x: len(x)) == 0, other=np.nan).rolling(windows_size, center = True).count()
    #errors = first_hour_chunk['duration'].where(first_hour_chunk['error'].apply(lambda x: len(x)) > 0, other=np.nan).rolling(windows_size, center = True).count()
    plt.stackplot(data['minute'],
                  data['successful_runs'],
                  data["failed_runs"],
                  labels=["Successfull workloads", "Failed workloads"])
    plt.ylabel("Amount of workloads executed")
    plt.xlabel("Scenario duration after rejuvenation (min)")
    plt.legend()
    plt.grid()
    fig = plt.gcf()
    fig.savefig('fig/HA_16_after_rejuv.pdf')
    plt.show()

def plot_HA_conc64():
    s = get_experiment_data_bars('high-availability/', 64, iteration=1, window=20)
    #cut out 5 hours of first run
    s[0] = (s[0][0][0:24], {node: frame[0:24] for node, frame in s[0][1].items()})
    rally_dfs = list()
    metric_dfs = list()
    controller_node = list(s[0][1].keys())[0]
    for (rally_data, metric_data) in s:
        rally_dfs.append(rally_data)
        metric_dfs.append(metric_data[controller_node])

    duration_change = get_bar_change(list(map(lambda data: data[0]['duration'], s)), s[0][0]['duration'][0])
    success_rate_change = get_bar_change(list(map(lambda data: data[0]["successful_runs"], s)), s[0][0]['successful_runs'][0])
    plt.figure().set_figwidth(12)
    for i in range(len(s)):
        time = s[i][0]['hour']
        if s[i][0]['hour'][0]>26.5:
            time = s[i][0]['hour'] - (s[i][0]['hour'][0]-26.5)
        plt.bar(time, duration_change[i], label="_" * i + "Average workload duration", color='C0')
        markerline, stemlines, baseline = plt.stem(time, success_rate_change[i], label="_" * i + "Amount of successfull workload executions", linefmt="C3")
        plt.setp(stemlines, linewidth=2)
    plt.ylabel(f"Relative to the start of the {scenario_word} (%)")
    plt.xlabel(scenario_duration)
    plt.legend(loc='upper center', bbox_to_anchor=(0.6, 1))
    plt.grid()
    fig = plt.gcf()
    fig.savefig('fig/HA_64_dur_succ.pdf')
    plt.show()

    db = get_experiment_data_bars('high-availability/', 64, iteration=1, window=20)
    #cut out 5 hours of first run
    db[0] = (db[0][0][0:24], {node: frame[0:24] for node, frame in db[0][1].items()})
    rally_dfs = list()
    metric_dfs = list()
    controller_node = list(db[0][1].keys())[0]
    for (rally_data, metric_data) in db:
        rally_dfs.append(rally_data)
        metric_dfs.append(metric_data[controller_node])
    plt.figure().set_figwidth(12)
    #all_runs = db[0][0]['successful_runs'] + db[0][0]['failed_runs']
    #succ_runs_rel = db[0][0]['successful_runs'] / all_runs
    #fail_runs_rel = db[0][0]['failed_runs'] / all_runs
    metr = 'hour'
    db[1][0]['hour'] = db[1][0]['hour'] - (db[1][0]['hour'][0] - 26.5)
    time = pd.concat([
        db[0][0][metr],
        pd.DataFrame([db[0][0][metr].iloc[-1]+1, db[1][0][metr][0]-1]),
        db[1][0][metr],
        pd.DataFrame([db[1][0][metr][0] + 1]),
        #db[2][0][metr],
        #pd.DataFrame([db[2][0][metr][0] + 1]),
        #db[3][0][metr]
    ],
        ignore_index=True)[0]
    metr = "successful_runs"
    succ = pd.concat([
        db[0][0][metr],
        pd.DataFrame([0, 0]),
        db[1][0][metr],
        pd.DataFrame([0]),
        #db[2][0][metr],
        #pd.DataFrame([0]),
        #db[3][0][metr]],
    ],
        ignore_index=True)[0]
    metr = 'failed_runs'
    fails = pd.concat([
        db[0][0][metr],
        pd.DataFrame([0, 0]),
        db[1][0][metr],
        pd.DataFrame([0]),
        #db[2][0][metr],
        #pd.DataFrame([0]),
        #db[3][0][metr]],
    ],
        ignore_index=True)[0]

    plt.stackplot(time, succ, fails, labels=["Successfull workloads", "Failed workloads"])
    plt.ylabel("Amount of workloads executed per hour")
    plt.xlabel(scenario_duration)
    plt.legend()
    plt.grid()
    fig = plt.gcf()
    fig.savefig('fig/HA_64_succ_fail.pdf')
    plt.show()

    m16 = get_experiment_metric('high-availability/', 16, iteration=1, window=20)
    m64 = get_experiment_metric('high-availability/', 64, iteration=1, window=20)
    control_df16 = m16[list(m16.keys())[0]]
    control_df64 = m64[list(m64.keys())[0]]

    control_df16 = control_df16[~((control_df16['hour'] > 26) & (control_df16['hour'] < 28))]
    control_df64 = control_df64[~((control_df64['hour'] > 26) & (control_df64['hour'] < 32))]

    second_part = control_df16['hour'][control_df16['hour']>26]
    control_df16['hour'][control_df16['hour']>26] = second_part - (second_part.min() - 26.5)
    skip = control_df16.loc[1555]
    control_df16 = control_df16[control_df16['hour']<28]
    control_df16.loc[1555] = skip
    control_df16 = control_df16.sort_index()
    second_part = control_df64['hour'][control_df64['hour']>26]
    control_df64['hour'][control_df64['hour']>26] = second_part - (second_part.min() - 26.5)
    #control_df64 = control_df64[control_df64['hour']<27.5]

    plt.figure().set_figwidth(12)

    plt.plot(control_df16['hour'], control_df16["node_memory_available_bytes_node_memory_MemAvailable_bytes"] / pow(2, 30),
             ls = "--", marker="x", markevery=20, label="Scenario#5, RAM available", color="C0")
    plt.plot(control_df16['hour'], control_df16["node_memory_swap_used_bytes"] / pow(2, 30),
             ls= '--', marker="x", markevery=20, label="Scenario#5, Swap used", color="C2")

    plt.axvline(x=1, color='r', ls = '--', label='Scenario#5, OpenStack fails')

    plt.plot(control_df64['hour'], control_df64["node_memory_available_bytes_node_memory_MemAvailable_bytes"] / pow(2, 30),
             label="Scenario#6, RAM available", color="C1")
    plt.plot(control_df64['hour'], control_df64["node_memory_swap_used_bytes"] / pow(2, 30), label=scenario_word+"#6, Swap used", color="C6")
    plt.axvline(x=14, color='r', label='Scenario#6, OpenStack fails')

    plt.grid()
    plt.xlabel(scenario_duration)
    plt.ylabel("Average memory usage (GB)")
    plt.legend(loc='upper center', bbox_to_anchor=(0.29, 1))
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig('fig/HA_16_64_memory.pdf')
    plt.show()

if not os.path.exists("fig"):
    os.makedirs("fig")
plt.rcParams['font.size'] = 16
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.figsize'][0] = 12
plot_HA_conc64()