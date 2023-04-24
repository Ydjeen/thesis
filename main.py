import itertools
import math
import os

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
    exp_folder = data_folder+structure+'/'+conc_fold+str(concurrency)+"/deploy1/rally/"
    if not os.path.isdir(exp_folder):
        exp_folder = data_folder + structure + '/' + conc_fold + str(concurrency) + f"/deploy_list/deploy{iteration}/rally/"
    task_folders = next(os.walk(exp_folder))[1]
    task_folders = ['{0}/{1}'.format(exp_folder, subfold) for subfold in task_folders]
    task_folders.sort()
    return task_folders

def get_metrics_folders(structure, concurrency):
    if not ensure_exp_folder(structure, concurrency):
        raise Exception(f'No experience folder provided for {structure} concurrency {concurrency}')
    exp_folder = data_folder+structure+'/'+conc_fold+str(concurrency)+"/deploy1/requests/"
    if not os.path.isdir(exp_folder):
        exp_folder = data_folder + structure + '/' + conc_fold + str(concurrency) + "/deploy_list/deploy1/requests/"
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

def extract_metrics(structure, concurrency):

    metric_names = None
    task_folders = get_task_folders(structure, concurrency)
    metrics_folders = get_metrics_folders(structure, concurrency)
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


def get_experiment_data_bars(struct, conc, iteration, window):
    rally_data_all = extract_rally_output(struct, conc, iteration)
    rally_data_all = list(map(lambda execution: pd.DataFrame(execution), rally_data_all))
    for data in rally_data_all:
        data.columns = ['timestamp', 'duration', 'error', 'actions']
        data['timestamp'] = data['timestamp'].replace(r'^\s*$', np.nan, regex=True).astype('float')
        data['duration'] = data['duration'].replace(r'^\s*$', np.nan, regex=True).astype('float')
    metrics_data_all = extract_metrics(struct, conc)
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
        if not timestamp_start:
            timestamp_start = timestamp_range[0]
        #reset timestmap start point
        rally_data['timestamp'] = rally_data['timestamp'] - timestamp_start
        for node, df in metric_data.items():
            df['timestamp'] = df['timestamp'] - timestamp_start

        bar_rally_data = {}
        total_chunks = int(round(((timestamp_range[1] - timestamp_range[0])/3600)))

        bar_metric_data = {}
        for node in metric_data:
            bar_metric_data[node] = pd.DataFrame()
        bar_time = list()
        bar_duration = list()
        bar_successfull_runs = list()
        bar_failed_runs = list()
        for i in range(total_chunks):
            chunk_start = rally_data['timestamp'].min() + (i * 3600)
            chunk_end = rally_data['timestamp'].min() + ((i+1) * 3600)
            chunk = rally_data[rally_data['timestamp'].between(chunk_start, chunk_end)]
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

            for node in bar_metric_data:
                chunk_avg = metric_data[node][metric_data[node]['timestamp'].between(chunk_start, chunk_end)].mean()
                bar_metric_data[node] = pd.concat((bar_metric_data[node], pd.DataFrame(chunk_avg).T), ignore_index=True)
        bar_rally_data = pd.DataFrame()
        bar_rally_data['timestamp'] = bar_time
        bar_rally_data['hour'] = (((bar_rally_data['timestamp']) / 3600) - 0.5).round() + 0.5
        bar_rally_data['duration'] = bar_duration
        bar_rally_data['successful_runs'] = bar_successfull_runs
        bar_rally_data['failed_runs'] = bar_failed_runs
        if not initial_performance:
            initial_performance = bar_duration[0]
        bar_rally_data['performance_change'] = (bar_rally_data['duration'] - initial_performance) / (
                initial_performance / 100) + 100
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
    plt.show()
    for request_bar in request_df_bar_list:
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


def plot_metrics(struct, conc, metric, iteration = 1, window = 20, node_to_plot = None):
    metrics = extract_metrics(struct, conc)
    metric_dfs = {}
    for run in metrics:
        for node, metrics in run.items():
            if node_to_plot:
                if node not in node_to_plot:
                    continue
            if node in metric_dfs.keys():
                temp = pd.concat([pd.DataFrame({"timestampt":[np.nan]}), pd.DataFrame(metrics)], ignore_index=True)
                metric_dfs[node] = pd.concat([metric_dfs[node], temp], ignore_index=True)
            else:
                metric_dfs[node] = pd.DataFrame(metrics)
    metric_to_parse = metric


    cleaning = False
    if cleaning:
        empty_index = list()
        for i, s in enumerate(metric_dfs[0]["wally190"][metric_to_parse]):
            if len(s) == 0:
                empty_index.append(i)
        print(f'For metric {metric} the following indexes are empty : {empty_index}')
        for node, metrics in before.items():
            for metric, values in metrics.items():
                eliminated = 0
                for index in empty_index:
                    values.pop(index - eliminated)
                    eliminated = eliminated + 1
    for node, df in metric_dfs.items():
        df['timestamp'] = df['timestamp'].astype('float')

    timestamp_start = metric_dfs[list(metric_dfs.keys())[0]]['timestamp'].min()
    for node, df in metric_dfs.items():
        metric_dfs[node] = metric_dfs[node].replace(r'^\s*$', np.nan, regex=True)
        metric_dfs[node]['hour'] = (df['timestamp'] - timestamp_start)/3600
        metric_dfs[node] = metric_dfs[node].astype('float')

    for node, df in metric_dfs.items():
        plt.plot(df['hour'], df[metric_to_parse], label=node)
    plt.title(metric_to_parse)
    plt.show()

    initial_performance = None
    request_df_bar_list = {}
    for node, df in metric_dfs.items():

        bar_time = list()
        bar_metric = list()
        bar_successfull_runs = list()
        bar_failed_runs = list()
        chunk_amount = int(((df['timestamp'].max() - df['timestamp'].min()) / 3600).round())
        for i in range(chunk_amount):
            chunk_start = df['timestamp'].min() + (i * 3600)
            chunk_end = df['timestamp'].min() + ((i+1) * 3600)
            chunk = df[df['timestamp'].between(chunk_start, chunk_end)]
            bar_time.append(chunk['timestamp'].mean())
            value = chunk[metric_to_parse].mean()
            if math.isnan(value):
                value = 0
            bar_metric.append(value)
        bar_data = pd.DataFrame()
        bar_data['timestamp'] = bar_time
        bar_data['hour'] = (((bar_data['timestamp']-timestamp_start)/3600)-0.5).round() + 0.5
        bar_data['value'] = bar_metric
        if not initial_performance:
            initial_performance = bar_metric[0]
        bar_data['performance_change'] = (bar_data['value'] - initial_performance)/(initial_performance/100) + 100

        request_df_bar_list[node] = bar_data

    for node, bar_data in request_df_bar_list.items():
        plt.stem(bar_data['hour'], bar_data['value'])
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

def get_bar_change(full_data):
    initial_value = full_data[0][0]
    change = list()
    for data in full_data:
        change.append((data - initial_value) / (initial_value / 100) + 100)
    return change

def plot_bars(s, metric):
    for (rally_data, metric_data) in s:
        for node in metric_data:
            plt.bar(rally_data['hour'], metric_data[node][metric])
    plt.show()
    controller_node = list(s[0][1].keys())[0]
    duration_performance_change = get_bar_change(list(map(lambda data: data[0]['duration'], s)))
    swap_amount_change = get_bar_change(list(map(lambda data: data[1][controller_node][metric], s)))
    for i in range(len(s)):
        plt.bar(s[i][0]['hour'], swap_amount_change[i], label="_" * i + "Swap usage change")
        plt.bar(s[i][0]['hour'], duration_performance_change[i], label= "_"*i + "WL duration change")
    plt.legend()
    plt.show()

#analyze_all(30)
struct = all_in_one_fld
conc = 8
iterations = [2]
window = 20

bars = True
dur = False
metr = True

if bars:
    for iteration in iterations:
        s = get_experiment_data_bars(struct, conc, iteration=iteration, window=window)
        plot_bars(s, 'node_memory_swap_used_bytes')

if dur:
    for iteration in iterations:
        plot_durations(struct, conc, iteration=iteration, window=window)

#plot_durations_for_actions(struct, conc, window, ['nova.boot'])

#metrics=['node_memory_available_bytes_node_memory_MemAvailable_bytes', 'node_memory_swap_used_bytes', 'node_cpu_utilisation_avg', 'available_space_/dev/mapper/vg0-root_ext4_/']
metrics=['node_memory_swap_used_bytes']

if metr:
    print_metric_names(struct, conc)
    metric = get_metric_list(struct, conc)[0]
    for iteration in iterations:
        for metric in metrics:
            plot_metrics(struct, conc, metric, iteration=iteration, window = window, node_to_plot="wally194")

