import pandas as pd
import numpy as np


def BaseFramen_CA():
    calendar = pd.read_csv('calendar.csv')
    drop_features = ['wm_yr_wk', 'wday',
                     'month', 'year', 'd', 'snap_TX', 'snap_WI', 'event_name_1', 'event_name_2']
    calendar = calendar.drop(drop_features, axis=1)
    calendar = calendar.set_index('date')
    type1 = calendar.event_type_1.unique()[1:]
    type2 = calendar.event_type_2.unique()[1:]
    type1 = pd.get_dummies(calendar.event_type_1)
    type2 = pd.get_dummies(calendar.event_type_2)
    type1.Cultural += type2.Cultural
    type1.Religious += type2.Religious
    w_day = pd.get_dummies(calendar.weekday)
    calendar = calendar.drop(
        ['event_type_1', 'event_type_2', 'weekday'], axis=1)
    BaseFrame = pd.concat([calendar, w_day, type1], axis=1)
    return BaseFrame
