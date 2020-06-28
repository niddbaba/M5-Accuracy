import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def CORR(shape, columns, data_sales):

    # 对每一家商店对每一个种类，只需要计算一个corr
    # 或者直接全部计算出来，然后在需要的时候调用
    corr = pd.DataFrame(data=np.full((shape, shape), np.nan),
                        columns=columns, index=columns)
    for col1 in columns:
        for col2 in columns:
            corr.loc[col1, col2] = pearsonr(
                data_sales[col1], data_sales[col2])[0]
    return corr


def max_corr_item(item, corr):
    return corr.index[corr[item] == max(corr[item])][0]


def min_corr_item(item, corr):
    return corr.index[corr[item] == min(corr[item])][0]
