import itertools
import math
import os
import re

import matplotlib
import numpy as np
import json
from matplotlib import pyplot as plt
import csv
import zipfile
import pandas as pd
from numpy import ndarray

data_folder = 'experiment_results/'
high_avail_fld = 'high-availability/'
all_in_one_fld = 'all-in-one'

node_memory_avg_free = "node_memory_free_relative"
nn = "node_avg_total_free_memory"
used = "node_memory_used_bytes"

conc_fold = "conc"

colors = []

def transpose_list(l):
    return list(map(list, zip(*l)))

def unzip_folder(input, output):
    with zipfile.ZipFile(input, 'r') as zip_ref:
        zip_ref.extractall(output)

def ensure_exp_folder(structure, concurrency):
    folder_needed = data_folder + structure + "/" + conc_fold + str(concurrency)
    if not os.path.isdir(folder_needed):
        if os.path.isfile(folder_needed+".zip"):
            unzip_folder(folder_needed+".zip", data_folder + structure + "/")
    if not os.path.isdir(folder_needed):
        print("no folder", folder_needed)
        return False
    return True

def extract_json_data(report_json):
    task_data = report_json['tasks'][0]['subtasks'][0]['workloads'][0]
    return task_data

def get_task_data_list(task_folders):
    result = list()
    for task_folder in task_folders:
        load_folders = next(os.walk(task_folder))[1]
        load_folders.sort()
        for load_folder in load_folders:
            report_json_file = open(task_folder+'/'+load_folder+'/'+'rally_report.json')
            report_json = json.load(report_json_file)
            result.append(extract_json_data(report_json))
    return result

def get_task_folders(structure, concurrency, iteration=1):
    if not ensure_exp_folder(structure, concurrency):
        raise Exception(f'No experience folder provided for {structure} concurrency {concurrency}')
    exp_folder = data_folder+structure+'/'+conc_fold+str(concurrency)+f"/deploy{iteration}/rally/"
    if not os.path.isdir(exp_folder):
        exp_folder = data_folder + structure + '/' + conc_fold + str(concurrency) + f"/deploy_list/deploy{iteration}/rally/"
    task_folders = next(os.walk(exp_folder))[1]
    task_folders = ['{0}/{1}'.format(exp_folder, subfold) for subfold in task_folders]
    task_folders.sort()
    return task_folders

def get_metrics_folders(structure, concurrency, iteration=1):
    if not ensure_exp_folder(structure, concurrency):
        raise Exception(f'No experience folder provided for {structure} concurrency {concurrency}')
    exp_folder = data_folder+structure+'/'+conc_fold+str(concurrency)+f"/deploy{iteration}/requests/"
    if not os.path.isdir(exp_folder):
        exp_folder = data_folder + structure + '/' + conc_fold + str(concurrency) + f"/deploy_list/deploy{iteration}/requests/"
    task_folders = next(os.walk(exp_folder))[1]
    task_folders = ['{0}/{1}'.format(exp_folder, subfold) for subfold in task_folders]
    task_folders.sort()
    return task_folders


def extract_rally_output(structure, concurrency, iteration=1):
    task_folders = get_task_folders(structure, concurrency, iteration)
    task_data_all = get_task_data_list(task_folders)
    data = list()
    tasks = sorted(task_data_all, key=lambda d: d['start_time'])
    for task_data in tasks:
        workload_data = list()
        for workload_info in task_data['data']:
            workload_data.append([workload_info['timestamp'], workload_info['duration']
                             ,workload_info['error'], workload_info['atomic_actions']])
        data.append(workload_data)
    return data

def extract_metrics(structure, concurrency, iteration = 1):

    metric_names = None
    task_folders = get_task_folders(structure, concurrency, iteration)
    metrics_folders = get_metrics_folders(structure, concurrency, iteration)
    metrics_folders.sort()
    all_results = list()
    for task_folder in metrics_folders:
        result = dict()
        load_folders = next(os.walk(task_folder))[1]
        load_folders.sort()
        for node in load_folders:
            with open(task_folder+'/'+node+'/custom_metrics.csv', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                metric_names = reader.__next__()
                metrics = list()
                last_time = None
                for row in reader:
                    if last_time is not None:
                        if last_time > float(row[0]):
                            break
                    metrics.append(row)
                    last_time = float(row[0])
                metrics = list(map(list, itertools.zip_longest(*metrics, fillvalue=None)))
                metrics = dict(zip(metric_names, metrics))
                result[node.split('.')[0]] = metrics
        all_results.append(result)
    # discards no data if jagged and fills short nested lists with None
    return all_results

def SMA(data, width):
    i = 0
    moving_averages = []
    for i in range(len(data)):
        start = 0
        if i < width/2:
            start = 0
            end = width/2 + i*2
        elif (i > len(data) - width/2):
            start = len(data) - width/2 + ((i - len(data) - width/2)/2)
            end = len(data)
        else:
            start = i - width/2
            end = i + width/2
        start = int(start)
        end = int(end)
        window = data[start:end]
        window_average = round(np.sum(window)/(end-start), 2)
        moving_averages.append(window_average)
    return moving_averages


def SMA_taken(data, width):
    i = 0
    moving_averages = []
    while i < len(data) - width + 1:
        window = data[i : i + width]
        window_average = round(np.sum(window) / width, 2)
        moving_averages.append(window_average)
        i += 1
    return moving_averages

def get_chunk_amount(df: pd.DataFrame):
    return round((df.timestamp.max() - df.timestamp.min()) / 3600)


def metrics_data_to_df(metrics_data):
    metric_dfs = {}
    for run in metrics:
        for node, metrics in run.items():
            if node_to_plot:
                if node not in node_to_plot:
                    continue
            if node in metric_dfs.keys():
                temp = pd.concat([pd.DataFrame({"timestampt": [np.nan]}), pd.DataFrame(metrics)], ignore_index=True)
                metric_dfs[node] = pd.concat([metric_dfs[node], temp], ignore_index=True)
            else:
                metric_dfs[node] = pd.DataFrame(metrics)


def get_experiment_data_bars(struct, conc, iteration, window=1, chunk_size = 3600):
    rally_data_all = extract_rally_output(struct, conc, iteration)
    rally_data_all = list(map(lambda execution: pd.DataFrame(execution), rally_data_all))
    for data in rally_data_all:
        data.columns = ['timestamp', 'duration', 'error', 'actions']
        data['timestamp'] = data['timestamp'].replace(r'^\s*$', np.nan, regex=True).astype('float')
        data['duration'] = data['duration'].replace(r'^\s*$', np.nan, regex=True).astype('float')
    metrics_data_all = extract_metrics(struct, conc, iteration)
    all_runs = list()
    timestamp_start = None
    initial_performance = None
    for i in range(max(len(rally_data_all), len(metrics_data_all))):
        rally_data = None
        metric_data = None
        rally_timestamp_range = None
        metrics_timestamp_range = None
        if len(rally_data_all) > i:
            rally_data = rally_data_all[i]
            rally_timestamp_range = (rally_data['timestamp'].min(), rally_data['timestamp'].max())

        if len(metrics_data_all) > i:
            metric_data = metrics_data_all[i]
            metric_dfs = {}
            for node, data in metric_data.items():
                metric_dfs[node] = pd.DataFrame(data).replace(r'^\s*$', np.nan, regex=True).astype('float')
                if not metrics_timestamp_range:
                    metrics_timestamp_range = (metric_dfs[node]['timestamp'].min(), metric_dfs[node]['timestamp'].max())
            metric_data = metric_dfs
        timestamp_range = rally_timestamp_range
        if not timestamp_range:
            timestamp_range = metrics_timestamp_range
        if not timestamp_start:
            timestamp_start = timestamp_range[0]
        #reset timestmap start point
        rally_data['timestamp'] = rally_data['timestamp'] - timestamp_start
        if metric_data:
            for node, df in metric_data.items():
                df['timestamp'] = df['timestamp'] - timestamp_start

        bar_rally_data = {}
        total_chunks = int(round(((timestamp_range[1] - timestamp_range[0])/chunk_size)))

        bar_metric_data = {}
        if metric_data:
            for node in metric_data:
                bar_metric_data[node] = pd.DataFrame()
        bar_time = list()
        bar_duration = list()
        bar_successfull_runs = list()
        bar_failed_runs = list()
        start_indexes = list()
        end_indexes = list()
        chunk_correction = 0.5
        chunk_time_shift = 1
        if chunk_size == 300:
            chunk_correction = 2.5
        for i in range(total_chunks):
            chunk_start = rally_data['timestamp'].min() + (i * chunk_size)
            chunk_end = rally_data['timestamp'].min() + ((i+1) * chunk_size)
            chunk = rally_data[rally_data['timestamp'].between(chunk_start, chunk_end)]
            start_indexes.append(chunk.index[0])
            end_indexes.append(chunk.index[-1])
            bar_time.append(chunk['timestamp'].mean())
            errors = chunk.apply(lambda x: True if x['error'] else False, axis=1)
            success = ~errors
            dur = chunk['duration'][success].mean()
            if math.isnan(dur):
                dur = 0
            ##TODO IF dur = float.nan not working
            bar_duration.append(dur)
            bar_successfull_runs.append(len(chunk[success]))
            bar_failed_runs.append(len(chunk[errors]))
            if metric_data:
                for node in bar_metric_data:
                    chunk_avg = metric_data[node][metric_data[node]['timestamp'].between(chunk_start, chunk_end)].mean()
                    chunk_avg['hour'] = (((chunk_avg['timestamp']) / chunk_size) - chunk_correction).round() + chunk_time_shift
                    bar_metric_data[node] = pd.concat((bar_metric_data[node], pd.DataFrame(chunk_avg).T), ignore_index=True)

        bar_rally_data = pd.DataFrame()
        bar_rally_data['timestamp'] = bar_time
        bar_rally_data['hour'] = (((bar_rally_data['timestamp']) / chunk_size) - chunk_correction).round() + chunk_time_shift
        bar_rally_data['duration'] = bar_duration
        bar_rally_data['successful_runs'] = bar_successfull_runs
        bar_rally_data['failed_runs'] = bar_failed_runs
        if not initial_performance:
            initial_performance = bar_duration[0]
        bar_rally_data['performance_change'] = (bar_rally_data['duration'] - initial_performance) / (
                initial_performance / 100) + 100
        bar_rally_data['start_index'] = start_indexes
        bar_rally_data['end_index'] = end_indexes
        all_runs.append((bar_rally_data, bar_metric_data))
    return all_runs

def plot_durations(struct, concurrency, iteration=1, window=1, bar=False):
    rally_data = extract_rally_output(struct, concurrency, iteration)

    requests_bar = list()
    requests_df = list(map(lambda execution: pd.DataFrame(execution), rally_data))

    request_df_bar_list = list()
    timestamp_start = requests_df[0][0].min()
    initial_performance = None
    for request in requests_df:
        request.columns=['timestamp', 'duration', 'error', 'actions']
        chunk_amount = get_chunk_amount(request)
        chunk_size = round(len(request) / get_chunk_amount(request))

        bar_time = list()
        bar_duration = list()
        bar_successfull_runs = list()
        bar_failed_runs = list()

        for i in range(get_chunk_amount(request)):
            chunk_start = request['timestamp'].min() + (i * 3600)
            chunk_end = request['timestamp'].min() + ((i+1) * 3600)
            chunk = request[request['timestamp'].between(chunk_start, chunk_end)]
            bar_time.append(chunk['timestamp'].mean())
            errors = chunk.apply(lambda x: True if x['error'] else False, axis=1)
            success = ~errors
            dur = chunk['duration'][success].mean()
            if math.isnan(dur):
                dur = 0
            ##TODO IF dur = float.nan not working
            bar_duration.append(dur)
            bar_successfull_runs.append(len(chunk[success]))
            bar_failed_runs.append(len(chunk[errors]))
        bar_data = pd.DataFrame()
        bar_data['timestamp'] = bar_time
        bar_data['hour'] = (((bar_data['timestamp']-timestamp_start)/3600)-0.5).round() + 0.5
        bar_data['duration'] = bar_duration
        bar_data['successful_runs'] = bar_successfull_runs
        bar_data['failed_runs'] = bar_failed_runs
        if not initial_performance:
            initial_performance = bar_duration[0]
        bar_data['performance_change'] = (bar_data['duration'] - initial_performance)/(initial_performance/100) + 100
        request_df_bar_list.append(bar_data)
    min = 9999
    for request_bar in request_df_bar_list:
        plt.bar(request_bar['hour'], request_bar['duration'])
        curr_min = request_bar['duration'][request_bar['duration'] > 0].min()
        if curr_min < min:
            min = curr_min
    if not min == 9999:
        plt.ylim(bottom = min*(2/3))
    plt.ylabel("Average workload duration (sec)")
    plt.xlabel("Experiment duration (hour)")
    plt.grid()
    plt.legend()
    plt.show()
    case = ["Before rejuvenation", "After 1st rejuvenation", "After 2nd rejuvenation", "After 3rd rejuvenation"]
    for (i, request_bar) in enumerate(request_df_bar_list):
        plt.bar(request_bar['hour'], request_bar['failed_runs'])
    plt.ylabel("Failed workloads")
    plt.xlabel("Experiment duration (hour)")
    plt.show()
    for request_bar in request_df_bar_list:
        plt.bar(request_bar['hour'], request_bar['successful_runs'])
    plt.ylabel("Successful workloads")
    plt.xlabel("Experiment duration (hour)")
    plt.show()
    for request_bar in request_df_bar_list:
        plt.bar(request_bar['hour'], request_bar['performance_change'])
    plt.ylabel("Performance change (%), relative to the first hour")
    plt.xlabel("Experiment duration (hour)")
    plt.show()
    return
    before_df = pd.DataFrame({'timestamp': before_data[0],'duration': before_data[1]})
    after_df = pd.DataFrame({'timestamp': after_data[0],'duration': after_data[1]})
    avg_before_df = before_df['duration'].rolling(window=window).mean()
    avg_after_df = after_df['duration'].rolling(window=window).mean()
    bar_before_df = list(map(np.mean, np.array_split(before_df['duration'], 24)))
    bar_after_df = list(map(np.mean, np.array_split(after_df['duration'], 1)))
    bar_time_before = list(map(np.mean, np.array_split(time_before, 24)))
    bar_time_after = list(map(np.mean, np.array_split(time_after, 1)))

    #avg_before = SMA(before_data[1], window)
    #avg_after = SMA(after_data[1], window)

    full_errors = before_data[2] + after_data[2]
    full_timesmtamps = np.concatenate((time_before, time_after))
    error_indexes = [i for i, x in enumerate(full_errors) if len(x)>0]

    error_groups = list()

    for index in error_indexes:
        if not error_groups:
            group = list()
            group.append(index)
            error_groups.append(group)
        else:
            if index - error_groups[-1][-1] <= concurrency and not (error_groups[-1][-1] < len(time_before) and index >= len(time_before)):
                error_groups[-1].append(index)
            else:
                group = list()
                group.append(index)
                error_groups.append(group)

    error_time_list = list()
    for error_group in error_groups:
        error_time_list.append([full_timesmtamps[index] for index in error_group])

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    ax.plot(time_before, avg_before_df, label="Before rejuvenation")
    ax.plot(time_after, avg_after_df, label="After rejuvenation")
    for i, error_time_group in enumerate(error_time_list):
        ax.axvspan(min(error_time_group), max(error_time_group), alpha=0.3, color='red', label= "_"*i + "Error")
    ax.plot()
    #plt.plot(time_before[window//2 : -window//2], avg_before[window//2 : -window//2], label="Before rejuvenation")
    #plt.plot(time_after[window//2 : -window//2], avg_after[window//2 : -window//2], label="After rejuvenation")
    plt.ylabel('Workload execution time (sec)')
    plt.xlabel('Experiment duration (sec)')
    plt.legend()
    plt.title("Durations")
    plt.show()

    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    plt.bar(bar_time_before, bar_before_df, width=2000, label="Before rejuvenation")
    plt.bar(bar_time_after, bar_after_df, width=2000, label="After rejuvenation")
    plt.ylim(min((bar_before_df + bar_after_df)) * (9/10))
    for i, error_time_group in enumerate(error_time_list):
       ax.axvspan(min(error_time_group), max(error_time_group), alpha=0.3, color='red', label="_" * i + "Error")
    # plt.plot(time_before[window//2 : -window//2], avg_before[window//2 : -window//2], label="Before rejuvenation")
    # plt.plot(time_after[window//2 : -window//2], avg_after[window//2 : -window//2], label="After rejuvenation")
    plt.ylabel('Workload execution time (sec)')
    plt.xlabel('Experiment duration (sec)')
    plt.legend()
    plt.title("bars")
    plt.show()

def plot_durations_for_actions(struct, concurrency, window, to_plot):
    rally_data = extract_rally_output(struct, concurrency)
    before_data = transpose_list(rally_data[0])
    after_data = transpose_list(rally_data[1])

    #rally_data = (rally_data)

    time_before = np.array(before_data[0]).astype('float')
    experiment_start = np.array(time_before[0]).astype('float')
    time_before = np.array(time_before)
    time_after = np.array(after_data[0])

    actions = [action['name'] for action in before_data[3][0]]
    before_actions = pd.DataFrame(before_data[3])
    after_actions = pd.DataFrame(after_data[3])
    #before_actions['timestamp'] = time_before
    #after_actions['timestamp'] = time_after


    before_df = pd.DataFrame({'timestamp': before_data[0],'duration': before_data[1]})
    after_df = pd.DataFrame({'timestamp': after_data[0],'duration': after_data[1]})
    avg_before_df = before_df['duration'].rolling(window=window).mean()
    avg_after_df = after_df['duration'].rolling(window=window).mean()

    #avg_before = SMA(before_data[1], window)
    #avg_after = SMA(after_data[1], window)

    full_durations = pd.concat((avg_before_df, avg_after_df))
    full_errors = before_data[2] + after_data[2]
    full_timesmtamps = np.concatenate((time_before, time_after))
    error_indexes = [i for i, x in enumerate(full_errors) if len(x)>0]

    error_groups = list()

    for index in error_indexes:
        if not error_groups:
            group = list()
            group.append(index)
            error_groups.append(group)
        else:
            if index - error_groups[-1][-1] <= concurrency and not (error_groups[-1][-1] < len(time_before) and index >= len(time_before)):
                error_groups[-1].append(index)
            else:
                group = list()
                group.append(index)
                error_groups.append(group)

    error_time_list = list()
    for error_group in error_groups:
        error_time_list.append([full_timesmtamps[index] for index in error_group])

    fig, ax = plt.subplots()
    fig.set_figwidth(15)
    not_print = 'delete'
    for module in to_plot:
        action_indexes_to_print = [index for index, action in enumerate(actions) if (module in action) and (not_print not in action)]
        for i in action_indexes_to_print:
            values = before_actions[i].apply(pd.Series)
            action_durations = values['finished_at'] - values['started_at']
            ax.plot(time_before, action_durations.rolling(window=window).mean(), label=values['name'][0])

            values = after_actions[i].apply(pd.Series)
            action_durations = values['finished_at'] - values['started_at']
            ax.plot(time_after, action_durations.rolling(window=window).mean(), label=values['name'][0])
    for i, error_time_group in enumerate(error_time_list):
        ax.axvspan(min(error_time_group), max(error_time_group), alpha=0.3, color='red', label= "_"*i + "Error")
    ax.plot()
    #plt.plot(time_before[window//2 : -window//2], avg_before[window//2 : -window//2], label="Before rejuvenation")
    #plt.plot(time_after[window//2 : -window//2], avg_after[window//2 : -window//2], label="After rejuvenation")
    plt.ylabel('Workload execution time (sec)')
    plt.xlabel('Experiment duration (sec)')
    plt.legend()
    plt.show()

def convertable_to_float(string):
    try:
        result = float(string)
        return True
    except ValueError:
        return False

def get_metric_list(config, conc):
    metrics = extract_metrics(config, conc)
    first_node = list(metrics[0].keys())[0]
    return list(metrics[0][first_node].keys())



def print_metric_names(config, conc):
    metric_list = get_metric_list(config, conc)
    last_metric = None
    to_print = ""
    for index, metric in enumerate(metric_list):
        if last_metric:
            if len(metric.split("_")) > 3 and metric.split("_")[:3] == last_metric.split("_")[:3]:
                to_print = f'{to_print}{index}:{metric}  '
            else:
                print(to_print)
                to_print = f'{index}:{metric}  '
        else:
            to_print = f'{index}:{metric}  '
        last_metric = metric
    print(to_print)


def get_error_times(struct, concurrency):
    rally_data = extract_rally_output(struct, concurrency)
    before_data = transpose_list(rally_data[0])
    after_data = transpose_list(rally_data[1])
    full_errors = before_data[2] + after_data[2]

    time_before = np.array(before_data[0]).astype('float')
    experiment_start = np.array(time_before[0]).astype('float')
    time_before = np.array(time_before)
    time_after = np.array(after_data[0])

    full_timesmtamps = np.concatenate((time_before, time_after))
    error_indexes = [i for i, x in enumerate(full_errors) if len(x) > 0]

    error_groups = list()

    for index in error_indexes:
        if not error_groups:
            group = list()
            group.append(index)
            error_groups.append(group)
        else:
            if index - error_groups[-1][-1] <= concurrency and not (
                    error_groups[-1][-1] < len(time_before) and index >= len(time_before)):
                error_groups[-1].append(index)
            else:
                group = list()
                group.append(index)
                error_groups.append(group)

    error_time_list = list()
    for error_group in error_groups:
        error_time_list.append([full_timesmtamps[index] for index in error_group])
    return error_time_list


def get_experiment_metric(struct, conc, iteration = 1, window = 20):
    metrics = extract_metrics(struct, conc, iteration)
    metric_dfs = {}
    for run in metrics:
        for node, metrics in run.items():
            if node in metric_dfs.keys():
                temp = pd.concat([pd.DataFrame({"timestampt":[np.nan]}), pd.DataFrame(metrics)], ignore_index=True)
                metric_dfs[node] = pd.concat([metric_dfs[node], temp], ignore_index=True)
            else:
                metric_dfs[node] = pd.DataFrame(metrics)

    for node, df in metric_dfs.items():
        df['timestamp'] = df['timestamp'].astype('float')

    timestamp_start = metric_dfs[list(metric_dfs.keys())[0]]['timestamp'].min()
    for node, df in metric_dfs.items():
        metric_dfs[node] = metric_dfs[node].replace(r'^\s*$', np.nan, regex=True)
        metric_dfs[node]['hour'] = (df['timestamp'] - timestamp_start)/3600
        metric_dfs[node] = metric_dfs[node].astype('float')
    return metric_dfs

def plot_basic_memory_info(metrics):
    control_df = m[list(m.keys())[0]]
    plt.plot(control_df['hour'], control_df["node_memory_available_bytes_node_memory_MemAvailable_bytes"]/pow(2, 30), label="RAM available")
    plt.plot(control_df['hour'], control_df["node_memory_swap_used_bytes"]/pow(2, 30), label="Swap used")
    plt.grid()
    plt.xlabel("Experiment duration (hour)")
    plt.ylabel("(GB)")
    plt.legend()
    plt.show()

def plot_metrics(x, y, label=None, xlabel=None, ylabel=None, title=None):
    plt.plot(x, y, label=label)
    plt.xlabel=xlabel
    plt.ylabel=ylabel
    plt.grid()
    plt.title(title)
    plt.show()
    return


    for node, metrics in before.items():
        metric_before = pd.DataFrame(before[node][metric_to_parse]).astype(float).rolling(window=window).mean()
        metric_after = pd.DataFrame(after[node][metric_to_parse]).astype(float).rolling(window=window).mean()
        time_before = np.array(before[node]["timestamp"]).astype(float)
        time_after = np.array(after[node]["timestamp"]).astype(float)
        #plt.plot(np.concatenate((time_before, time_after)), np.concatenate((metric_before, metric_after), axis=0), label=node)
        plt.plot(np.concatenate((time_before, time_after)), pd.concat((metric_before, metric_after)), label=node)

    rally_data = extract_rally_output(struct, conc)
    before_data = transpose_list(rally_data[0])
    after_data = transpose_list(rally_data[1])
    before_start = before_data[0][0]
    before_end = before_data[0][-1]
    after_start = after_data[0][0]
    after_end = after_data[0][-1]
    plt.axvspan(before_start, before_end, alpha=0.05, color='blue', label="First run")
    plt.axvspan(after_start, after_end, alpha=0.05, color='green', label="Second run")
    error_groups = get_error_times(struct, conc)
    for i, error_time_group in enumerate(error_groups):
        plt.axvspan(min(error_time_group), max(error_time_group), alpha=0.3, color='red', label="_" * i + "Error")
    plt.title(metric_to_parse)
    plt.legend(loc='lower center')
    plt.show()

def analyze_run(struct, conc, window):
    [before, after] = extract_rally_output(struct, conc)
    analyzed_data = {}
    analyzed_data['starting_avg_dur'] = round(np.average([i[1] for i in before[conc:conc+window]]))
    analyzed_data['middle_avg_dur'] = round(np.average([i[1] for i in before[int((len(before)-window)/2):int((len(before)+window)/2)]]))
    analyzed_data['ending_avg_dur'] = round(np.average([i[1] for i in before[-(conc+window):-conc]]))
    if window < len(after)-(conc*2):
        analyzed_data['reset_avg_dur'] = round(np.average([i[1] for i in after[conc:conc+window]]))
    else:
        analyzed_data['reset_avg_dur'] = round(np.average([i[1] for i in after[conc:-conc]]))
    print(len(after))
    return analyzed_data

def analyze_all(window):
    struct = high_avail_fld
    runs = {}
    for conc in [1, 2, 4]:
        runs[conc] = analyze_run(struct, conc, window)

    for conc, info in runs.items():
        print(f"Concurrency {conc}")
        print(f" start:mid:end:reset avg duration ")
        print(f" {info['starting_avg_dur']} : {info['middle_avg_dur']} : {info['ending_avg_dur']} : {info['reset_avg_dur']} ")

def get_bar_change(full_data, initial_value=None):
    if not initial_value:
        initial_value = full_data[0][0]
    change = list()
    for data in full_data:
        change.append((data - initial_value) / (initial_value / 100) + 100)
    return change

def plot_bars(s, metric):
    rally_dfs = list()
    metric_dfs_controller = list()
    metric_dfs = list()
    controller_node = list(s[0][1].keys())[0]
    for (rally_data, metric_data) in s:
        rally_dfs.append(rally_data)
        if metric_data:
            metric_dfs_controller.append(metric_data[controller_node])
            metric_dfs.append(metric_data)
    duration_performance_change = get_bar_change(list(map(lambda data: data[0]['duration'], s)))
    swap_amount_change = get_bar_change(list(map(lambda data: data[controller_node][metric], metric_dfs)))
    corellation = pd.DataFrame({"dur_change": duration_performance_change[0], "swap_change": swap_amount_change[0]}).corr()
    for i in range(len(s)):
        if i < len(swap_amount_change):
            plt.stem(s[i][0]['hour'], swap_amount_change[i], label="_" * i + "Swap usage change")
        if i < len(duration_performance_change):
            plt.stem(s[i][0]['hour'], duration_performance_change[i], label= "_"*i + "WL duration change", markerfmt="red")
    plt.legend()
    plt.title(f"Corellation = {corellation}")
    plt.show()

    rally_df = pd.concat(rally_dfs, ignore_index=True)
    metric_df = pd.concat(metric_dfs_controller, ignore_index=True)
    duration_diff_pct = rally_df.pct_change()['duration']
    swap_diff_pct = metric_df.pct_change()[metric]
    corellation = pd.DataFrame({"dur_change": duration_diff_pct, "swap_change": swap_diff_pct}).corr()['swap_change']['dur_change']
    plt.stem(rally_df['hour'][:len(duration_diff_pct)], duration_diff_pct, label="Swap usage change")
    plt.stem(rally_df['hour'][:len(swap_diff_pct)], swap_diff_pct, label= "WL duration change", markerfmt="red")
    plt.legend()
    plt.title(f"Corellation = {corellation}")
    plt.show()

def error_to_human(error_list):
    if len(error_list) == 0:
        return None
    if re.compile('Quota exceeded for instances').search(error_list[2]):
        return "NOVA: Quota exceeded for instance"
    if re.compile('executer._rebuild_server').search(error_list[2]):
        return "NOVA: Rebuild server error"
    if re.compile('Quota exceeded.*router').search(error_list[2]):
        return "NEUTRON: Quota exceeded for router"
    if re.compile('Quota exceeded.*security_group').search(error_list[2]):
        return "NEUTRON: Quota exceeded for security group"
    if re.compile(' Maximum number of volumes allowed \(10\) exceeded for quota').search(error_list[2]):
        return "CINDER: Quota exceeded for volume"
    if re.compile('Unable to establish connection to http://130.149.249').search(error_list[2]):
        if "admin_keystone.create_user()" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable create user"
        if "add_role(user_id=self.executer.user.id" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable add role"
        if "_create_security_group" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable security group"
        if "revoke_role" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable revoke role"
        if "delete_user" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable delete user"
        if "delete_role" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable delete_role"
        if "detach_volume" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable detach_volume"
        if "create_role" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable create_role"
        if "attach_volume" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable attach_volume"
        return "CONNECTION_ERROR: Another node is unreachable create_user"
    if re.compile('http://130.149.249.*timed out').search(error_list[2]):
        if "admin_keystone.create_user()" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable create user"
        if "add_role(user_id=self.executer.user.id" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable add role"
        if "_create_security_group" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable security group"
        if "revoke_role" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable revoke role"
        if "delete_user" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable delete user"
        if "delete_role" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable delete_role"
        if "detach_volume" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable detach_volume"
        if "create_role" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable create_role"
        if "attach_volume" in error_list[2]:
            return "CONNECTION_ERROR: Another node is unreachable attach_volume"
        return "CONNECTION_ERROR: Another node is unreachable"
    if re.compile('self.executer.create_volume_params').search(error_list[2]):
        return "CINDER: Create volume"
    if re.compile('Invalid volume: Volume status must be').search(error_list[2]):
        return "CINDER: Invalid volume"
    if re.compile('Cannot.*detach.*while it is in task_state rebuild').search(error_list[2]):
        return "NOVA: Detach volume"
    if re.compile('Cannot.*detach.*while it is in vm_state error').search(error_list[2]):
        return "NOVA: Detach volume"
    if re.compile('self.executer.server_kwargs').search(error_list[2]):
        return "NOVA: Create server"
    if re.compile('download.cirros-cloud.net.*Network is unreachable').search(error_list[2]):
        return "NETWORK_ERROR: Unable to download image"
    if re.compile('cirros-0.3.5-x86_64-disk.img.*Connection refused').search(error_list[2]):
        return "NETWORK_ERROR: Unable to download image"
    if re.compile('cirros-0.3.5-x86_64-disk.img.*Connection timed out').search(error_list[2]):
        return "NETWORK_ERROR: Unable to download image"
    if re.compile('cirros-0.3.5-x86_64-disk.img.*No route to host').search(error_list[2]):
        return "NETWORK_ERROR: Unable to download image"
    print(error_list[2])
    raise Exception(error_list)


def get_error_info(task_data, concurrency):
    task_data = pd.DataFrame(task_data[0], columns= ["timestamp", "duration", "error", "atomic_actions"])
    error = task_data['error']
    error_old = error
    error_formated = error.apply(lambda x: error_to_human(x))
    error_info = {}
    amount_of_errors_per_wl = error_formated.apply(lambda x: x == None)
    failed_workloads = error_formated[~error_formated.isnull()]
    amount_of_errors = failed_workloads.index.size
    success_workloads = error_formated[error_formated.isnull()]
    amount_of_success = success_workloads.index.size
    last_success_index = success_workloads.index[-1]
    fail_point = last_success_index

    failed = False

    first_error = None
    first_error_index = None
    last_error_before_fail = None
    last_error_before_fail_index = None
    error_amount_before_fail = None
    success_amount_before_fail = None
    first_error_after_fail = None
    first_error_after_fail_index = None
    last_error_amount_before_fail = None

    if amount_of_errors > 0:
        if last_success_index+concurrency <= failed_workloads.index[-1]:
            failed = True
    errors_before_fail = None
    errors_after_fail = None
    if failed_workloads.size > 0:
        first_error_index = failed_workloads.index[0]
        first_error = failed_workloads[first_error_index]
    if failed:
        errors_before_fail_index = failed_workloads.index[failed_workloads.index <= fail_point]
        errors_before_fail = failed_workloads[errors_before_fail_index]
        errors_after_fail_index = failed_workloads.index[failed_workloads.index >= fail_point]
        errors_after_fail = failed_workloads[errors_after_fail_index]

        last_error_before_fail_index = errors_before_fail_index[-1]
        last_error_before_fail = failed_workloads[last_error_before_fail_index]
        last_error_amount_before_fail = errors_before_fail[errors_before_fail==last_error_before_fail].count()

        error_amount_before_fail = errors_before_fail_index.size
        success_amount_before_fail = success_workloads.size

        first_error_after_fail_index = failed_workloads.index[failed_workloads.index > fail_point][0]
        first_error_after_fail = failed_workloads[first_error_after_fail_index]

    error_info.update({'first_error':first_error, 'first_error_index':first_error_index })
    error_info.update({'last_error_before_fail':last_error_before_fail, "last_error_before_fail_index":last_error_before_fail_index})
    error_info.update({'last_error_amount_before_fail': last_error_amount_before_fail})
    error_info.update({"error_amount_before_fail":error_amount_before_fail })
    error_info.update({"success_amount_before_fail":success_amount_before_fail})
    error_info.update({"first_error_after_fail":first_error_after_fail, "first_error_after_fail_index":first_error_after_fail_index})

    unique_errors = error_formated.dropna().unique()
    error_stat = pd.DataFrame()
    error_data = list()
    for unique_error in unique_errors:
        before_occ = None
        before_occ_perc = None
        after_occ = None
        after_occ_perc = None
        total_occ = error_formated[error_formated == unique_error].count()
        total_occ_perc = (total_occ/error_formated.count()) * 100
        if failed:
            before_occ = errors_before_fail[errors_before_fail==unique_error].count()
            before_occ_perc = (before_occ/errors_before_fail.count())*100
            after_occ = errors_after_fail[errors_after_fail==unique_error].count()
            after_occ_perc = (after_occ/errors_after_fail.count())*100
        curr_stat = {"Total occurence":total_occ, "Total occurence perc.":total_occ_perc,
                     "Before failed occurence": before_occ, "Before failed occurence perc.": before_occ_perc,
                     "After failed occurence": after_occ, "After failed occurence perc.": after_occ_perc}
        error_stat[unique_error] = curr_stat
    if failed:
        error_data = [errors_before_fail, errors_after_fail]
    else:
        error_data = [failed_workloads]
    return error_info, error_stat, error_data


def plot_error_data(HA_error_data, AiO_error_data):
    concurrency_list = [1, 2, 4, 8, 16, 64]
    AiO_iteration_to_use = [1, 4, 1, 4, 2, 1]

    exp6_bars = get_experiment_data_bars(high_avail_fld, 64, 1, 20)
    exp6_errors = HA_error_data[5]

    error = pd.concat([exp6_errors[0], exp6_errors[1]])
    unique_errors = error.dropna().unique()
    c = list()
    for er in unique_errors:
        if "uota exceeded" not in er:
            c.append(er)
    unique_errors = c

    bar_error = {}

    for unique_error in unique_errors:
        bar_error[unique_error] = list()

    for bar_index in exp6_bars[0][0].index:
        start_index = exp6_bars[0][0]['start_index'][bar_index]
        end_index = exp6_bars[0][0]['end_index'][bar_index]
        errors_in_bar = error.iloc[(error.index>=start_index) & (error.index<= end_index)]
        for unique_error in unique_errors:
            bar_error[unique_error].append(int(errors_in_bar[errors_in_bar==unique_error].count()))

    for unique_error in unique_errors:
        if sum(bar_error[unique_error]) == 0:
            continue
        plt.bar(exp6_bars[0][0]['hour'], bar_error[unique_error], label=unique_error)
        break
    plt.legend()
    plt.show()


def print_error_stat():
    concurrency_list = [1,2,4,8,16,64]
    AiO_iteration_to_use=[1,4,1,4,2,1]
    HA_error_info = pd.DataFrame()
    AiO_error_info = pd.DataFrame()
    HA_error_stats = list()
    AiO_error_stats = list()
    HA_error_data = list()
    AiO_error_data = list()
    cached = False
    if cached:
        HA_error_info = pd.read_csv("tmp/HA_error.csv", index_col=0)
        AiO_error_info = pd.read_csv("tmp/AiO_error.csv", index_col=0)
        for i in range(6):
            HA_error_stats.append(pd.read_csv(f"tmp/HA_stat{i}.csv", index_col=0))
            AiO_error_stats.append(pd.read_csv(f"tmp/AiO_stat{i}.csv", index_col=0))
    else:
        for scenario_counter in range(6):
            conc = concurrency_list[scenario_counter]
            HA_data = extract_rally_output(high_avail_fld, conc)
            AiO_data = extract_rally_output(all_in_one_fld, conc, AiO_iteration_to_use[scenario_counter])
            error_info, error_stats, error_data = get_error_info(HA_data, conc)
            HA_error_info = pd.concat([HA_error_info, pd.DataFrame([error_info])], ignore_index=True)
            HA_error_stats.append(error_stats)
            HA_error_data.append(error_data)
            error_info, error_stats, error_data = get_error_info(AiO_data, conc)
            AiO_error_info = pd.concat([AiO_error_info, pd.DataFrame([error_info])], ignore_index=True)
            AiO_error_stats.append(error_stats)
            AiO_error_data.append(error_data)

        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        HA_error_info.to_csv("tmp/HA_error.csv")
        AiO_error_info.to_csv("tmp/AiO_error.csv")
        for index in range(6):
            HA_error_stats[index].to_csv(f"tmp/HA_stat{index}.csv")
            AiO_error_stats[index].to_csv(f"tmp/AiO_stat{index}.csv")
    print("High availability data")
    print(HA_error_info.to_markdown())
    print("All-in-one data")
    print(AiO_error_info.to_markdown())
    config_list = ["Mult", "AiO"]
    concurrencies=[1,2,4,8,16,64]
    total_data = pd.DataFrame()
    for i, data in enumerate(HA_error_stats):
        if i > 2:
            total_data = total_data.add(data, fill_value=0)
    for i, data in enumerate(AiO_error_stats):
        if i > 2:
            total_data = total_data.add(data, fill_value=0)
    print(total_data.to_markdown())

    plot_error_data(HA_error_data, AiO_error_data)


    for config, data in enumerate([HA_error_stats, AiO_error_stats]):
        for index, stat in enumerate(data):
            if stat.empty:
                continue
            stat_t = stat.T
            before_dominance = stat_t[stat_t['Before failed occurence perc.'] > 50]
            if before_dominance.size > 0:
                stat_without_dom = stat_t
                stat_without_dom = stat_without_dom.drop(before_dominance.index)
                stat_without_dom = stat_without_dom[stat_without_dom['Before failed occurence'] != 0]
                values = stat_without_dom["Before failed occurence"]
                fig, ax = plt.subplots()
                pie = plt.pie(x=stat_without_dom["Before failed occurence"], shadow=True)
                total_errors = int(stat_without_dom["Before failed occurence"].sum())
                text = ax.text(1.3, 1.4, f"Excluded: {before_dominance.index[0]}: {round(before_dominance['Before failed occurence perc.'][0],2)}%"
                                       f"\nOther occurred errors: {total_errors}, among them:",  fontsize=14,
                        verticalalignment="top")
                leg = plt.legend(pie[0], map(lambda a,b: f"{a}\n{int(b)} occurrence{'s' if int(b)>1 else ''}" ,
                                             stat_without_dom.index, stat_without_dom["Before failed occurence"]),
                           bbox_to_anchor=(0.4, 0.8), loc="upper left", fontsize=14,
                           bbox_transform=plt.gcf().transFigure)
                plt.subplots_adjust(left=0.0, bottom=0.1, right=0.5)
                fig = plt.gcf()
                fig.savefig(f'fig/{config_list[config]}_{concurrencies[index]}_pie_failures.pdf')
                plt.show()

    for config, data in enumerate([HA_error_stats, AiO_error_stats]):
        for index, stat in enumerate(data):
            if stat.empty:
                continue
            stat_t = stat.T
            before_dominance = stat_t[stat_t['After failed occurence perc.'] > 50]
            if before_dominance.size > 0:
                stat_without_dom = stat_t
                stat_without_dom = stat_without_dom.drop(before_dominance.index)
                stat_without_dom = stat_without_dom[stat_without_dom['After failed occurence'] != 0]
                values = stat_without_dom["After failed occurence"]
                fig, ax = plt.subplots()
                pie = plt.pie(x=stat_without_dom["After failed occurence"], shadow=True)
                total_errors = int(stat_without_dom["After failed occurence"].sum())
                text = ax.text(1.3, 1.4, f"Excluded: {before_dominance.index[0]}: {round(before_dominance['After failed occurence perc.'][0],2)}%"
                                       f"\nOther occurred errors: {total_errors}, among them:",  fontsize=14,
                        verticalalignment="top")
                leg = plt.legend(pie[0], map(lambda a,b: f"{a}\n{int(b)} occurrence{'s' if int(b)>1 else ''}" ,
                                             stat_without_dom.index, stat_without_dom["After failed occurence"]),
                           bbox_to_anchor=(0.4, 0.8), loc="upper left", fontsize=14,
                           bbox_transform=plt.gcf().transFigure)
                plt.subplots_adjust(left=0.0, bottom=0.1, right=0.5)
                fig = plt.gcf()
                #fig.savefig(f'fig/{config_list[config]}_{concurrencies[index]}_pie_failures.pdf')
                plt.show()


def mann_kendall_test(data):
    mk_value = 0
    param = data[:].reset_index(drop=True)
    mean = 0
    for k in range(param.size-1):
        for l in range(k+1, param.size):
            diff = param[l] - param[k]
            if diff < 0:
                diff = -1
            if diff > 0:
                diff = 1
            mean = mean + diff
    ties = sum([(x.size)*((x.size)-1)*(2*(x.size)+5) for x in param.unique() if x.size>1])
    var = 1/18 * (param.size*(param.size - 1)*(2*param.size + 5) - ties)
    if mean == 0:
        mk_value = 0
    if mean > 0:
        mk_value = (mean-1)/math.sqrt(var)
    if mean < 0:
        mk_value = (mean+1)/math.sqrt(var)
    slope_all = list()
    for i in range(param.size-1):
        for j in range(i+1, param.size):
            diff = (param[j] - param[i])/(j-i)
            slope_all.append(diff)
    slope_all.sort()
    slope = 0
    try:
        slope = slope_all[int(len(slope_all) / 2)]
    except:
        slope = 0
    return mk_value, slope

def aging_rejuv(data, rejuv_data):
    param = data[:].reset_index(drop=True)
    return param.get(param.size-1) - param[0], param.get(param.size-1) - rejuv_data[0]

def get_duration_bar(bars):
    return bars[0][0]['duration']

def mann_kendall_test_all():
    concurrency_list = [1, 2, 4, 8, 16, 64]
    AiO_iteration_to_use = [1, 4, 1, 4, 2, 1]
    HA_iteration_to_use = [1,1,1,1,1,1]


    print("All in One data")
    print("RRT Mandall | Slope || Swap Mandall | Slope || Available Mandall | Slope")
    for i in range (6):
        bars = get_experiment_data_bars(all_in_one_fld, concurrency_list[i], AiO_iteration_to_use[i])
        duration_bar = bars[0][0]['duration']
        duration_bar = duration_bar[duration_bar > 0]

        swap_used_bar = next(iter(bars[0][1].values()))['node_memory_swap_used_bytes']/pow(2, 30)
        swap_used_bar = swap_used_bar[swap_used_bar > 0]

        memory_avail_bar = next(iter(bars[0][1].values()))['node_memory_available_bytes_node_memory_MemAvailable_bytes']/pow(2, 30)
        memory_avail_bar = memory_avail_bar[memory_avail_bar > 0]
        values = [i for sub in [mann_kendall_test(duration_bar), mann_kendall_test(swap_used_bar), mann_kendall_test(memory_avail_bar)] for i in sub]
        print(["{0:0.3f}".format(i*24) for i in values], f'Scen. {i+7}')

    print("Multinode data")
    for i in range (6):
        bars = get_experiment_data_bars(high_avail_fld, concurrency_list[i], HA_iteration_to_use[i])
        duration_bar = bars[0][0]['duration']
        duration_bar = duration_bar[duration_bar > 0]

        swap_used_bar = next(iter(bars[0][1].values()))['node_memory_swap_used_bytes']/pow(2, 30)
        swap_used_bar = swap_used_bar[swap_used_bar > 0]

        memory_avail_bar = next(iter(bars[0][1].values()))['node_memory_available_bytes_node_memory_MemAvailable_bytes']/pow(2, 30)
        memory_avail_bar = memory_avail_bar[memory_avail_bar > 0]

        values = [i for sub in [mann_kendall_test(duration_bar), mann_kendall_test(swap_used_bar),
                                mann_kendall_test(memory_avail_bar)] for i in sub]
        print(["{0:0.3f}".format(i*24) for i in values], f'Scen. {i+1}')

def direct_aging_rejuvenation():
    concurrency_list = [1, 2, 4, 8, 16, 64]
    AiO_iteration_to_use = [1, 4, 1, 4, 2, 1]
    HA_iteration_to_use = [1,1,1,1,1,1]


    print("All in One data")
    print("RRT Aging | Rejuv || Swap Aging | Rejuv || Available Aging | Rejuv")
    for i in range (6):
        bars = get_experiment_data_bars(all_in_one_fld, concurrency_list[i], AiO_iteration_to_use[i])
        duration_bar = bars[0][0]['duration']
        valuable_points = duration_bar > 0
        duration_bar = duration_bar[valuable_points]
        if valuable_points[valuable_points==True].size < 10:
            continue
        rejuv_duration = bars[1][0]['duration']


        swap_used_bar = next(iter(bars[0][1].values()))['node_memory_swap_used_bytes']/pow(2, 30)
        swap_used_bar = swap_used_bar[valuable_points]
        swap_used_rejuv = next(iter(bars[1][1].values()))['node_memory_swap_used_bytes'] / pow(2, 30)

        memory_avail_bar = next(iter(bars[0][1].values()))['node_memory_available_bytes_node_memory_MemAvailable_bytes']/pow(2, 30)
        memory_avail_bar = memory_avail_bar[valuable_points]
        memory_avail_rejuv = next(iter(bars[1][1].values()))[
                               'node_memory_available_bytes_node_memory_MemAvailable_bytes'] / pow(2, 30)

        values = [i for sub in [aging_rejuv(duration_bar, rejuv_duration), aging_rejuv(swap_used_bar, swap_used_rejuv), aging_rejuv(memory_avail_bar, memory_avail_rejuv)] for i in sub]
        print(["{0:0.3f}".format(i) for i in values], f'Scen. {i+7}')

    print("Multinode data")
    for i in range (6):
        bars = get_experiment_data_bars(high_avail_fld, concurrency_list[i], HA_iteration_to_use[i])
        duration_bar = bars[0][0]['duration']
        valuable_points = duration_bar > 0
        duration_bar = duration_bar[valuable_points]
        if valuable_points[valuable_points==True].size < 10:
            continue
        rejuv_duration = bars[1][0]['duration']

        if bars[1][1].values().__len__() == 0:
            continue
        swap_used_bar = next(iter(bars[0][1].values()))['node_memory_swap_used_bytes']/pow(2, 30)
        swap_used_bar = swap_used_bar[valuable_points]
        swap_used_rejuv = next(iter(bars[1][1].values()))['node_memory_swap_used_bytes'] / pow(2, 30)

        memory_avail_bar = next(iter(bars[0][1].values()))['node_memory_available_bytes_node_memory_MemAvailable_bytes']/pow(2, 30)
        memory_avail_bar = memory_avail_bar[valuable_points]
        memory_avail_rejuv = next(iter(bars[1][1].values()))[
                                 'node_memory_available_bytes_node_memory_MemAvailable_bytes'] / pow(2, 30)

        values = [i for sub in [aging_rejuv(duration_bar, rejuv_duration),
                                aging_rejuv(swap_used_bar, swap_used_rejuv), aging_rejuv(memory_avail_bar, memory_avail_rejuv)] for i in sub]
        print(["{0:0.3f}".format(i) for i in values], f'Scen. {i+1}')


if __name__ == "__main__":
    struct =  high_avail_fld
    conc = 16
    iterations = [1]
    window = 20

    bars = True
    dur = True
    metr = True

    plt.rcParams['font.size'] = 16
    plt.rcParams['figure.figsize'][0] = 12

    mann_kendall_test_all()
    exit()

    HA_data = extract_rally_output(all_in_one_fld, 2)
    temp = get_error_info(HA_data, 64)
    print(temp)
    print_error_stat()
    exit()

    if bars:
        for iteration in iterations:
            s = get_experiment_data_bars(struct, conc, iteration=iteration, window=window)
            plot_bars(s, 'node_memory_swap_used_bytes')

    if dur:
        for iteration in iterations:
            plot_durations(struct, conc, iteration=iteration, window=window)

    # plot_durations_for_actions(struct, conc, window, ['nova.boot'])
    # metrics=['node_memory_available_bytes_node_memory_MemAvailable_bytes', 'node_memory_swap_used_bytes', 'node_cpu_utilisation_avg', 'available_space_/dev/mapper/vg0-root_ext4_/']
    print_metric_names(struct, conc)
    metrics = ['node_memory_swap_used_bytes']

    if metr:
        for iteration in iterations:
            m = get_experiment_metric(struct, conc, iteration=iteration, window=window)
            plot_basic_memory_info(m)


