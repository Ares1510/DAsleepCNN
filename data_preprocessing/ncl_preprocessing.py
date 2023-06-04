import glob
import concurrent.futures
import pandas as pd
from agcounts.extract import get_counts

def agcount_algorithm(ax_data, freq=40, epoch=30):
    """
    This function is to convert the raw accelerometer data into activity counts.
    :param ax_data:  the raw accelerometer data
    :param freq:    the sampling frequency
    :param epoch:   the epoch length
    :return:    the activity counts
    """
    counts = get_counts(ax_data, freq=freq, epoch=epoch, fast=False, verbose=False)
    counts = pd.DataFrame(counts, columns=["Axis1", "Axis2", "Axis3"])
    counts["counts"] = (counts["Axis1"] ** 2 + counts["Axis2"] ** 2 + counts["Axis3"] ** 2
                        ) ** 0.5
    counts = counts.drop(columns=["Axis1", "Axis2", "Axis3"])
    return counts

def get_acc(subject_id):
    file_path = glob.glob(f'data/NCL/acc/*{subject_id}*left*.csv')[0]
    df = pd.read_csv(file_path)
    df.drop(df.columns[4:], axis=1, inplace=True)
    df.columns = ['x', 'y', 'z', 'timestamp']
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M:%S:%f')
    #downsample to 40Hz
    df = df.resample('25L', on='timestamp').mean()
    df = df.interpolate(method='linear')
    start_time = df.index[0]
    counts = agcount_algorithm(df.values, freq=40, epoch=30)
    time_series = pd.Series(pd.date_range(start=start_time, periods=len(counts), freq='30S'))
    #create dataframe with time_series as index and counts as the only column
    df = pd.DataFrame(counts.values, columns=['counts'], index=time_series.dt.time)
    return df

def get_psg(subject_id):
    file_path = glob.glob(f'data/NCL/psg/*{subject_id}*.txt')[0]
    df = pd.read_table(file_path, skiprows=17, sep='\t')
    df['Time [hh:mm:ss]'] = pd.to_datetime(df['Time [hh:mm:ss]'], format='%H:%M:%S')
    time_series = df['Time [hh:mm:ss]'].dt.time
    df.set_index(time_series, inplace=True)
    return df

def process_subject(subject_id, target_path):
    """Reads in a file and returns a dataframe"""
    counts_df = get_acc(subject_id)
    psg_df = get_psg(subject_id)
    df = psg_df.join(counts_df , how='inner')[['counts', 'Sleep Stage']]
    df['Sleep Stage'] = df['Sleep Stage'].map({'W': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'R': 4})
    df.to_csv(target_path + '.csv', index=False)

def main():
    subject_ids = ['01', '02', '10', '14', '17', '21', '23', '27', '28', '29']
    target_paths = [f'data/NCL/{x}' for x in subject_ids]

    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        executor.map(process_subject, subject_ids, target_paths)


if __name__ == '__main__':
    main()

