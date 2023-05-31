# **** Description *****
# This script will load PSG data and activity counts data, and then align them and merge them together.

import os

import numexpr
from joblib import Parallel
from joblib import delayed
# The activity counts data is from the following repo:
from agcounts.extract import get_counts
from activity_counts_functions import *
import pandas as pd


def get_all_files_include_sub(path, file_type):
    files = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if file_type in file[-len(file_type):]:
                files.append(os.path.join(os.path.abspath(r), file))
    return files


def agcount_algorithm(ax_data, freq=100, epoch=30):
    """
    This function is to convert the raw accelerometer data into activity counts.
    :param ax_data:  the raw accelerometer data
    :param freq:    the sampling frequency
    :param epoch:   the epoch length
    :return:    the activity counts
    """
    counts = get_counts(ax_data, freq=freq, epoch=epoch, fast=False, verbose=True)
    counts = pd.DataFrame(counts, columns=["Axis1", "Axis2", "Axis3"])
    counts["counts"] = (counts["Axis1"] ** 2 + counts["Axis2"] ** 2 + counts["Axis3"] ** 2
                        ) ** 0.5
    counts = counts.drop(columns=["Axis1", "Axis2", "Axis3"])
    return counts


def actigraph_lindert_counts(ax_data, freq, timezone):
    """
    This function is to convert the raw accelerometer data into activity counts.
    :param ax_data:     the raw accelerometer data
    :param freq:    the sampling frequency of the data
    :param timezone:    the timezone, e.g. UTC
    :return:    the activity counts
    """
    ax_data = up_down_sampling_and_fs_filtering(ax_data, sourceFs=freq, requiredFs=50)
    ax_data = build_activity_counts_without_matlab(ax_data, epoch=30)
    # convert to activity counts

    ax_data = pd.DataFrame(data=ax_data)
    ax_data.columns = ['time_stamp', 'counts']
    ax_data['time_stamp'] = pd.to_datetime(ax_data['time_stamp'], unit='s').dt.round('30S')
    ax_data['time_stamp'] = ax_data['time_stamp'].dt.tz_localize(timezone)
    ax_data = ax_data.dropna(how="all")
    return ax_data['counts']
    # combined_df = pd.concat((psg_df, ax['counts']), axis=1)
    #
    # # ts = ax[:, 0]
    # ts = df['Time'].dt.round('30S')
    # ts = ts.unique()
    # ts = ts[0: counts.shape[0]]
    # ts = pd.DataFrame(ts, columns=["time_stamp"])

def process_a_file(psg_f, all_csvs, output_path, count_method):
    """
    given a psg file, find the corresponding activity counts file, and then merge them together
    :param psg_f:  the psg file
    :param all_csvs:  all the activity counts files
    :param output_path:  the output path
    :param count_method:  the count algorithm to use, e.g. lindert or agcount
    :return:
    """
    pid = psg_f.split(os.sep)[-1].split("-")[0]
    psg_file_name = psg_f.split(os.sep)[-1].split(".")[0]

    psg_df = pd.read_csv(psg_f)
    # psg_df['time_stamp'] = pd.to_datetime(psg_df['time_stamp']).dt.tz_convert('UTC')
    psg_df['time_stamp'] = pd.to_datetime(psg_df['time_stamp'])
    timezone = psg_df['time_stamp'][0].tzinfo
    for x in all_csvs:
        if (pid in x) and ('100hz' in x):
            df = pd.read_csv(x)
            df = df[['Time', 'Accel-X (g)', ' Accel-Y (g)', ' Accel-Z (g)']]
            df['Time'] = pd.to_datetime(df['Time'])
            df['Time'] = df['Time'].dt.tz_localize(timezone)
            # df['Time'] = df.dt.tz_convert('UTC')
            # PSG annotation is made per 30s, so the first epoch is actually started 30s ago
            # Cut the data into the start and end of the PSG annotation

            df = df[(df['Time'] >= psg_df['time_stamp'].iloc[0] - pd.Timedelta(30, 's')) &
                    (df['Time'] <= psg_df['time_stamp'].iloc[-1])]

            activity_count_output_path = os.path.join(output_path, psg_file_name + "-activity_counts.csv")
            # parse the time stamp
            ax = df.values
            ax[:, 0] = np.round(
                (df['Time'].dt.tz_localize(None) - pd.Timestamp("1970-01-01")) / pd.Timedelta(1000, 'ms'), 0)
            if count_method == "agcounts":
                # counts = count_func(ax[:, 1:], freq=100, epoch=30, fast=False, verbose=True)
                counts = agcount_algorithm(ax[:, 1:], freq=100, epoch=30)
            elif count_method == "lindert":
                counts = actigraph_lindert_counts(ax, freq=100, timezone=timezone)
            else:
                raise ValueError("Unknown count algorithm")

            combined_df = pd.concat([psg_df, counts], axis=1)
            combined_df.to_csv(activity_count_output_path, index=False)

if __name__ == "__main__":
    count_alg_name = "lindert"
    csv_root = r"P:\IDEA-FAST\_data\S-8921602b"
    all_csvs = get_all_files_include_sub(csv_root, ".csv")
    psg_root = r"P:\IDEA-FAST\_data\PSG\PSG_Annotation\annotations"
    psg_annotations = get_all_files_include_sub(psg_root, ".csv")
    output_path = fr"P:\IDEA-FAST\_data\PSG\rana_activity_{count_alg_name}"
    parallel = False
    if not parallel:
        for i in np.arange(0, len(psg_annotations)):
            process_a_file(psg_annotations[i], all_csvs, output_path, count_alg_name)
    else:
        executor = Parallel(n_jobs=numexpr.detect_number_of_cores() - 5, backend='multiprocessing')
        # create tasks so we can execute them in parallel
        tasks = (delayed(process_a_file)(psg_annotations[i], all_csvs, output_path, count_alg_name) for i in np.arange(0, len(psg_annotations)))
        # execute tasks
        executor(tasks)
