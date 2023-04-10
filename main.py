import itertools
import os

import matplotlib
import numpy as np
import json
from matplotlib import pyplot as plt
import csv
import zipfile

data_folder = 'experiment_results/'
high_avail_fld = 'high-availability/'
all_in_one_fld = 'all-in-one'

node_memory_avg_free = "node_memory_free_relative"
nn = "node_avg_total_free_memory"
used = "node_memory_used_bytes"

conc_fold = "conc"

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
        exp_folder = data_folder + structure + conc_fold + str(concurrency) + "/deploy_list/deploy1/rally/"
    task_folders = next(os.walk(exp_folder))[1]
    task_folders = ['{0}/{1}'.format(exp_folder, subfold) for subfold in task_folders]
    task_folders.sort()
    return task_folders

def get_metrics_folders(structure, concurrency):
    if not ensure_exp_folder(structure, concurrency):
        raise Exception(f'No experience folder provided for {structure} concurrency {concurrency}')
    exp_folder = data_folder+structure+'/'+conc_fold+str(concurrency)+"/deploy1/requests/"
    if not os.path.isdir(exp_folder):
        exp_folder = data_folder + structure + conc_fold + str(concurrency) + "/deploy_list/deploy1/requests/"
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
                for row in reader:
                    metrics.append(row)
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
            end = width
        elif (i > len(data) - width/2):
            start = len(data) - width/2
            end = len(data)
        else:
            start = i - width/2
            end = i + width/2
        start = int(start)
        end = int(end)
        window = data[start:end]
        window_average = round(np.sum(window)/width, 2)
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
    avg_before = SMA(before_data[1], window)
    time_before = matplotlib.dates.date2num(before_data[0])
    avg_after = SMA(after_data[1], window)
    time_after = matplotlib.dates.date2num(after_data[0])
    plt.figure().set_figwidth(15)
    plt.plot(time_before, avg_before)
    plt.plot(time_after, avg_after)
    plt.show()

def convertable_to_float(string):
    try:
        result = float(string)
        return True
    except ValueError:
        return False

def plot_metrics(config, conc):
    metrics_HA = extract_metrics(config, conc)
    before = metrics_HA[0]
    after = metrics_HA[1]
    plt.figure().set_figwidth(15)
    metric_to_parse = list(before[list(before.keys())[0]].keys())[-10]
    print(metric_to_parse)
    empty_index = list()
    for i, s in enumerate(before["wally190"][metric_to_parse]):
        if len(s) == 0:
            empty_index.append(i)
    print(f'empty indexes : {empty_index}')

    cleaning = False
    if cleaning:
        for node, metrics in before.items():
            for metric, values in metrics.items():
                eliminated = 0
                for index in empty_index:
                    values.pop(index - eliminated)
                    eliminated = eliminated + 1

    time_before = before[list(before.keys())[0]]["timestamp"]
    time_after = after[list(before.keys())[0]]["timestamp"]
    for node, metrics in before.items():
        metric_before = np.array(before[node][metric_to_parse]).astype(float)
        metric_after = np.array(after[node][metric_to_parse]).astype(float)
        plt.plot(time_before + time_after, np.concatenate((metric_before, metric_after), axis=0), label=node)
    plt.title(metric_to_parse)
    plt.legend()
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
#plot_durations(high_avail_fld, 1, 50)
plot_metrics(high_avail_fld, 1)

