import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import BaseFrame as bf
import corr as cr
import keras
import MyLSTMModel


# 找到第一个销量非0时间
def start_index(frame):
    for index, row in frame.iterrows():
        if row['self_sales'] > 0:
            return index


def ResetFrame(sales, BF):
    data = np.concatenate((sales.values, np.full((1969-1913,), np.nan)))
    return data


prices_CA = pd.read_csv('prices_CA.csv')
data = pd.read_csv('sales_train_validation.csv')
submission = pd.read_csv('sample_submission.csv')
# 结果DataFrame


# 准备加州的数据数据
stores_CA = data.loc[data.state_id == 'CA'].store_id.unique()
for store in stores_CA:  # 对每个加州对商城遍历
    data_store = data.loc[data.store_id == store]

    cats = data_store.cat_id.unique()
    for cat in cats:   # 对每一个该商城对种类遍历
        data_store_cat = data_store.loc[data_store.cat_id == cat]
        data_sales = data_store_cat[data_store_cat.columns[6:]].T
        data_sales.columns = data_store_cat.id
        # 计算商品的关联性性质
        corr = pd.read_csv('corr.csv', index_col='id')

        for item in data_sales.columns:
            '''
            对每一个商品, 首先建立其BaseFrame, 再加入自己的价格, 加入最大和最小关联商品的价格因素
            最后加入自己的商品的销量, 为test_y
            '''
            BaseFrame_CA = bf.BaseFramen_CA()
            BaseFrame_CA['self_prices'] = prices_CA[item].values

            max_corr_item = cr.max_corr_item(item, corr)
            min_corr_item = cr.min_corr_item(item, corr)

            BaseFrame_CA['max_corr_prices'] = prices_CA[max_corr_item].values
            BaseFrame_CA['min_corr_prices'] = prices_CA[min_corr_item].values
            BaseFrame_CA['self_sales'] = ResetFrame(
                data_sales[item], BaseFrame_CA)

            start_index = start_index(BaseFrame_CA)
            train_data = BaseFrame_CA.loc[start_index:]
            # print(train_data)
            # print(train_data.shape)
            res = MyLSTMModel.train_and_predict(train_data)  # （B，T，F）
            print(res)
            print(res.shape)

            # 划分训练集和测试集
