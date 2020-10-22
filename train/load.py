from datetime import datetime
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.utils import shuffle


def process_eod_data(data_path, seq=16, tra_end=5200, val_end=5900):
    samples = [sample for sample in os.listdir(data_path) if # '../data/limit_daily/SSE'
               os.path.isfile(os.path.join(data_path, sample))]
    print(len(samples), 'samples selected')  # sample股票数

    min_max_scaler = MinMaxScaler()
    data_EOD = []
    for sample in samples:
        single_EOD = np.genfromtxt(os.path.join(data_path, sample), dtype=np.float32,
                                   delimiter=',', skip_header=False)
        single_EOD_scale = min_max_scaler.fit_transform(single_EOD)

        data_EOD.append(single_EOD_scale[:seq, :])

    data_EOD = np.array(data_EOD)

    limit = pd.read_csv('../data/limit_daily/limit_ratio_count_SH.csv')
    ground_truth = np.where(np.array(limit['ratio']) >= 0.05, 1, 0)[:, np.newaxis]

    data_EOD, ground_truth = shuffle(data_EOD, ground_truth, random_state=0)

    tra_pv = data_EOD[:tra_end,:,:]
    tra_gt = ground_truth[:tra_end,:]
    val_pv = data_EOD[tra_end:val_end,:,:]
    val_gt = ground_truth[tra_end:val_end,:]
    test_pv = data_EOD[val_end:,:,:]
    test_gt = ground_truth[val_end:,:]
    return tra_pv, tra_gt, val_pv, val_gt, test_pv, test_gt

# tra_pv, tra_gt, val_pv, val_gt, test_pv, test_gt=process_eod_data('',4766,5900)

def load_eod_data(data_path, tra_ind, val_ind, tes_ind):
    pass
