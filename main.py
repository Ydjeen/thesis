import itertools
import os

import matplotlib
import numpy as np
import json
from matplotlib import pyplot as plt
import csv
import zipfile
import pandas as pd

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

def get_task_folders(structure, concurrency):
    if not ensure_exp_folder(structure, concurrency):
        raise Exception(f'No experience folder provided for {structure} concurrency {concurrency}')
    exp_folder = data_folder+structure+'/'+conc_fold+str(concurrency)+"/deploy1/rally/"
    if not os.path.isdir(exp_folder):
        exp_folder = data_folder + structure + '/' + conc_fold + str(concurrency) + "/deploy_list/deploy1/rally/"
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


def extract_rally_output(structure, concurrency):
    task_folders = get_task_folders(structure, concurrency)
    task_data_all = get_task_data_list(task_folders)
    data = list()
    before = [task for task in task_data_all if task['runner']['constant_for_duration']['duration'] == 86400][0]
    after = [task for task in task_data_all if task['runner']['constant_for_duration']['duration'] != 86400][0]
    for task_data in [before, after]:
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

def plot_durations(struct, concurrency, window):
    rally_data = extract_rally_output(struct, concurrency)
    before_data = transpose_list(rally_data[0])
    after_data = transpose_list(rally_data[1])
    #rally_data = (rally_data)

    time_before = np.array(before_data[0]).astype('float')
    experiment_start = np.array(time_before[0]).astype('float')
    time_before = np.array(time_before)
    time_after = np.array(after_data[0])

    before_df = pd.DataFrame({'timestamp': before_data[0],'duration': before_data[1]})
    after_df = pd.DataFrame({'timestamp': after_data[0],'duration': after_data[1]})
    avg_before_df = before_df['duration'].rolling(window=window).mean()
    avg_after_df = after_df['duration'].rolling(window=window).mean()

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
    before = metrics[0]
    return list(before["wally190"].keys())


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


def plot_metrics(struct, conc, window, metric):
    metrics = extract_metrics(struct, conc)
    before = metrics[0]
    after = metrics[1]
    plt.figure().set_figwidth(15)
    metric_to_parse = metric
    empty_index = list()
    for i, s in enumerate(before["wally190"][metric_to_parse]):
        if len(s) == 0:
            empty_index.append(i)
    print(f'For metric {metric} the following indexes are empty : {empty_index}')

    cleaning = False
    if cleaning:
        for node, metrics in before.items():
            for metric, values in metrics.items():
                eliminated = 0
                for index in empty_index:
                    values.pop(index - eliminated)
                    eliminated = eliminated + 1

    for node, metrics in before.items():
        empty_indexes = [i for i, e in enumerate(before[node][metric_to_parse]) if e == '']
        for ind in empty_indexes:
            before[node][metric_to_parse][ind] = None
        empty_indexes = [i for i, e in enumerate(after[node][metric_to_parse]) if e == '']
        for ind in empty_indexes:
            after[node][metric_to_parse][ind] = None
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

#analyze_all(30)
plot_durations(all_in_one_fld, 1, 20)

#plot_durations_for_actions(high_avail_fld, 2, 20, ['nova.boot'])



#print_metric_names(all_in_one_fld, 1)
#metric = get_metric_list(all_in_one_fld, 1)[120]
#plot_metrics(all_in_one_fld, 1, 20, metric)