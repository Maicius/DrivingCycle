import pandas as pd
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics, cross_validation, ensemble, linear_model
from sklearn.preprocessing import scale, minmax_scale
import json
import xgboost as xgb
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GridSearchCV
from copy import deepcopy
import multiprocessing

from util import draw_double_line, plot_v_t, font

matplotlib.rcParams['axes.unicode_minus']=False

class DataProcess(object):
    raw_pre = 'raw_data/'
    mid_pre = 'mid_data/'
    image_pre = 'image/'
    result_pre = 'result_data/'
    idling_v_thresh = 10
    idling_time_thresh = 180
    MOVE_TIME = 1200
    FINAL_MOVEMENT = []
    miss_index = []
    PROCESS_NUM = 4
    move_feature = []
    movement_state = 0
    movement_state_1 = 0
    movement_state_2 = 0
    movement_state_3 = 0
    def __init__(self):
        pass

    def main(self):
        self.load_data()
        self.data1 = self.cal_acc_speed(self.data1)
        self.data1 = self.cal_idling_speed(self.data1)
        self.data1 = self.find_movement_state(self.data1)
        self.get_complete_true_movement(self.data1, index=1)

        print("预处理后的data 1大小：", self.data1.shape[0])
        # self.cal_movement_feature(self.data1)
        self.movement_state_1 = self.movement_state
        print("move state 1:", self.movement_state_1)
        self.data2 = self.cal_acc_speed(self.data2)
        self.data2 = self.cal_idling_speed(self.data2)
        self.data2 = self.find_movement_state(self.data2)
        self.get_complete_true_movement(self.data2, index=2)
        print("预处理后的data 2大小：", self.data2.shape[0])

        # self.cal_movement_feature(self.data1, 2)
        self.movement_state_2 = self.movement_state
        print("move state 2:", self.movement_state_2)
        self.data3 = self.cal_acc_speed(self.data3)
        self.data3 = self.cal_idling_speed(self.data3)
        self.data3 = self.find_movement_state(self.data3)
        self.get_complete_true_movement(self.data3, index=3)
        print("预处理后的data 3大小：", self.data3.shape[0])
        # self.load_miss_data()
        self.data = pd.concat([self.data1, self.data2, self.data3], axis=0)
        # # # self.data = self.data1
        # self.miss_data['abnormal'] = 1
        # self.miss_data = self.miss_data.loc[:, self.data.columns]
        # self.data = pd.concat([self.data, self.miss_data])
        self.cal_movement_feature(self.data)

    def load_miss_data(self):
        self.miss_data1 = pd.read_csv(self.mid_pre + "new_miss_data1.csv")
        self.miss_data2 = pd.read_csv(self.mid_pre + "new_miss_data2.csv")
        self.miss_data3 = pd.read_csv(self.mid_pre + "new_miss_data3.csv")
        self.miss_data = pd.concat([self.miss_data1, self.miss_data2, self.miss_data3])
        self.miss_data.reset_index(inplace=True)
        self.miss_data = self.cal_acc_speed(self.miss_data, final=True)

    def get_complete_true_movement(self, data, index=1):
        miss_movement = data.loc[data.order.isin(self.miss_index), 'movement']
        print("有异常的运动学片段: 数据集",index, len(set(miss_movement)))
        miss_data = data.loc[data.movement.isin(miss_movement)]
        miss_data.to_csv(self.mid_pre + "miss_data" + str(index) + ".csv")
        data.drop(data.loc[data.movement.isin(miss_movement)].index, inplace=True)
        print("完整的运动片段: 数据集",index, len(set(data.movement.values)))
        data.to_csv(self.mid_pre + "complete" + str(index) + ".csv")
        miss_index_df = pd.DataFrame(self.miss_index)
        miss_index_df.to_csv(self.mid_pre + "miss_index_df" + str(index) + ".csv", index=False)

    def load_data(self):
        self.data1 = pd.read_csv(self.raw_pre + "new1.csv")
        columns = ['order', 'time', 'gps_v', 'x', 'y', 'z', 'longitude', 'latitude', 'engine',
                              'torque', 'fuel', 'accelerator', 'air', 'engine_load', 'flow', 'a']
        self.data1.columns = columns

        self.data2 = pd.read_csv(self.raw_pre + "new2.csv")
        self.data2.columns = columns
        self.data3 = pd.read_csv(self.raw_pre + "new3.csv")
        self.data3.columns = columns

        self.data1['time'] = self.data1['time'].apply(lambda x: pd.to_datetime(x[:-4]))
        self.data2['time'] = self.data2['time'].apply(lambda x: pd.to_datetime(x[:-4]))
        self.data3['time'] = self.data3['time'].apply(lambda x: pd.to_datetime(x[:-4]))
        print("原始数据大小：", self.data1.shape[0], self.data2.shape, self.data3.shape)
        print("累计:", self.data1.shape[0] + self.data2.shape[0] + self.data3.shape[0])

    def find_waste_time_split(self, data, time_split_index):
        for index in time_split_index:
            v = data.loc[data.index == index, 'gps_v'].values[0]
            v1 = data.loc[data.index == index - 1, 'gps_v'].values[0]

            if v1 != 0 and v == 0:
                data.loc[data.index == index - 1, 'time_split'] = 2
                data.loc[data.index == index, 'time_split'] = 0
            elif v1 != 0 and v != 0:
                data.loc[data.index == index - 1, 'time_split'] = 2
                data.loc[data.index == index, 'time_split'] = 2
            elif v1 == 0 and v == 0:
                data.loc[data.index == index, 'time_split'] = 1
        return data

    def cal_acc_speed(self, data, final=False):
        v_data = data.loc[:, ['time', 'gps_v']]
        print("原始数据大小：", data.shape)
        if not final:
            v_data['time'] = v_data['time'].apply(lambda x: time.mktime(x.timetuple()))

        begin = pd.DataFrame([{'time': v_data.loc[0, 'time'] - 1, 'gps_v': 0}])
        length = v_data.shape[0]
        # print("data shape", v_data.shape)
        v_data2 = pd.concat([begin, v_data.loc[:length - 2, :]])
        # print("data2 shape", v_data2.shape)
        v_data2.columns = ['gps_v2', 'time2']
        # v_data的index可以作为之后筛选的id值
        v_data.reset_index(inplace=True)
        v_data2.reset_index(inplace=True)
        v_data2.drop(['index'], axis=1, inplace=True)
        v_data3 = pd.concat([v_data, v_data2], axis=1)

        v_data3['time_step'] = (v_data3['time'] - v_data3['time2'])
        time_split_index = v_data3.loc[v_data3['time_step'] >= 180, 'index']

        self.miss_index = v_data3.loc[(v_data3['time_step'] > 1) & (v_data3['time_step'] < 180), 'index']
        print("miss movement index", len(set(self.miss_index)))
        v_data3['time_split'] = 0
        v_data3.loc[v_data3.index.isin(time_split_index), 'time_split'] = 1
        v_data3 = self.find_waste_time_split(v_data3, time_split_index)
        # v_data3 = v_data3.loc[v_data3['time_step'] <= 1, :]

        print("去除不连续值:", v_data3.shape)
        # 计算加速度
        v_data3['acc'] = (v_data3['gps_v'] - v_data3['gps_v2']) / (v_data3['time_step'] * 3.6)

        max_acc = 100 / (7 * 3.6)
        min_acc = -8
        # print(v_data3.shape)
        print("去除异常加速度之前:", data.shape)
        miss_v_data3 = v_data3.loc[(v_data3['acc'] > max_acc) | (v_data3['acc'] < min_acc)]

        self.miss_index = self.miss_index.append(miss_v_data3.iloc[:, 0])
        # v_data3 = v_data3.loc[(v_data3['acc'] <= max_acc) & (v_data3['acc'] >= min_acc)]
        # data = data.loc[data.order.isin(v_data3.iloc[:, 0])]
        # data['acc'] = v_data3['acc']
        # print("去除异常加速度之后：", v_data3.shape)
        print(v_data3.shape, data.shape)
        data['time_split'] = v_data3['time_split']
        data['time_step'] = v_data3['time_step']

        data['acc'] = v_data3['acc']
        # print(self.miss_index)
        return data

    def get_multi_miss_data(self):

        for i in range(1, 4):
            miss_data = pd.read_csv(self.mid_pre + "miss_data" + str(i) + ".csv")
            miss_index = pd.read_csv(self.mid_pre + "miss_index_df" + str(i) + ".csv")
            complete_data = pd.read_csv(self.mid_pre + "complete" + str(i) + ".csv")
            self.find_knn_movement(index=i, miss_data=miss_data, miss_index=miss_index, complete_data=complete_data)

    def find_knn_movement(self, index, miss_data, miss_index, complete_data):
        # miss_data = pd.read_csv(self.mid_pre + "miss_data.csv")
        # miss_index = pd.read_csv(self.mid_pre + "miss_index_df.csv")
        miss_data['abnormal'] = 0
        miss_data_group = miss_data.groupby("movement")
        movement_ids = miss_data_group.movement.indices.keys()

        new_miss_move_list = []
        for movement_id in movement_ids:
            miss_move = miss_data_group.get_group(movement_id)
            shape = miss_move.shape[0]
            miss_move_list = miss_move.to_dict(orient='records')
            new_miss_move_list.append(miss_move_list[0])
            for i in range(1, shape):
                step = miss_move_list[i]['time'] - miss_move_list[i - 1]['time']
                if step > 1:
                    begin_time = miss_move_list[i - 1]['time'] + 1
                    for j in range(int(step) - 1):
                        new_miss_move_list.append(dict(time=begin_time, abnormal=1, movement=movement_id))
                        begin_time += 1
                new_miss_move_list.append(miss_move_list[i])
        new_miss_move_df = pd.DataFrame(new_miss_move_list)
        new_miss_move_df.fillna(0, inplace=True)
        new_miss_move_df.loc[miss_data.index.isin(miss_index['index']), 'abnormal'] = 1
        new_miss_move_df.to_csv(self.mid_pre + "miss_move_fill" + str(index) + ".csv")
        print("保存文件完成")
        data_group = complete_data.groupby('movement')
        com_movement_id = data_group.movement.indices.keys()

        com_shape_list = []
        for com_id in com_movement_id:
            df = data_group.get_group(com_id)
            shape = df.shape[0]
            com_shape_list.append(dict(length=shape, movement_id=com_id))
        com_shape_list = sorted(com_shape_list, key=lambda x: x['length'], reverse=True)

        new_miss_move_group = new_miss_move_df.groupby("movement")

        movement_ids = list(miss_data_group.movement.indices.keys())

        step = len(movement_ids) // self.PROCESS_NUM

        print(step)
        process_list = []
        print(len(movement_ids))
        for i in range(self.PROCESS_NUM):
            begin_index = i * step
            end_index = (i + 1) * step
            if i == self.PROCESS_NUM - 1:
                movement_temp = movement_ids[begin_index:]
            else:
                movement_temp = movement_ids[begin_index: end_index]
            print(begin_index, end_index)
            print(len(movement_temp))
            p = multiprocessing.Process(target=self.cal_knn_matrix, args=(movement_temp, new_miss_move_group, com_shape_list, data_group, i, step, index))
            p.daemon = True
            process_list.append(p)
        for p in process_list:
            p.start()

        for p in process_list:
            p.join()

        print("finish to cal knn")
        self.update_user_dict(index)

    def update_user_dict(self, index):
        all_user_dict = []
        for i in range(self.PROCESS_NUM):
            with open(self.mid_pre + "knn_result" + str(index) + str(i) + '.json', 'r') as r:
                data = json.load(r)
                all_user_dict.append(data)
        with open(self.mid_pre + "knn_result" + str(index) + ".json", 'w', encoding='utf-8') as w:
            json.dump(all_user_dict, w)

    def cal_knn_matrix(self, movement_ids, new_miss_move_group, com_shape_list, data_group, thread_id, step, index):
        columns = ['gps_v', 'engine', 'torque', 'fuel', 'accelerator', 'air', 'engine_load', 'flow', 'a']
        result_dict = []
        if thread_id == 0:
            print(thread_id)
            pass
        for miss_move_id in movement_ids:
            try:
                print("---------------------------")
                print("begin:", miss_move_id)
                move_df = new_miss_move_group.get_group(miss_move_id).reset_index()
                all_length = move_df.shape[0]
                if all_length < 10:
                    continue
                abnormal_index = move_df.loc[move_df['abnormal'] == 1, :].index
                if len(abnormal_index) == 0:
                    continue
                max_index = max(abnormal_index)
                min_index = min(abnormal_index)
                miss_len = len(abnormal_index)
                if miss_len >= all_length * 0.5:
                    print(thread_id, "缺失值太长，放弃。。。")
                    continue
                print("缺失值长度:", miss_len)
                if miss_len < 10:
                    half_window = 50
                else:
                    half_window = miss_len * 2
                max_index = min(max_index + half_window, all_length)
                min_index = max(min_index - half_window, 0)
                move_df = move_df.iloc[min_index: max_index, :]
                move_df = move_df.loc[:, columns]
                count = 0
                length = move_df.shape[0]
                print(move_df.shape)
                for com in com_shape_list:
                    print(count)
                    if com['length'] <= length:
                        print("异常的data df 较长")
                        break
                    else:
                        print("thread id", thread_id, count)
                        compare_data = data_group.get_group(com['movement_id'])
                        error_list = []
                        for i in range(0, com['length'] - length):
                            slide_window_df = compare_data.iloc[i: i + length, :]
                            slide_window_df = slide_window_df.loc[:, columns]

                            compare_error = pd.DataFrame(
                                np.subtract(minmax_scale(slide_window_df.values), minmax_scale(move_df.values)))
                            compare_error = compare_error.apply(lambda x: x ** 2)
                            compare_error['sum_val'] = compare_error.sum(axis=1)
                            # compare_error['sum_val'] = np.sqrt(compare_error['sum_val'])
                            # result_error = compare_error['sum_val'].sum()
                            # compare_error['sum_val'] = compare_error['sum_val'].apply(lambda x: x**2)
                            result_error2 = np.sqrt(compare_error['sum_val'].sum())
                            error_list.append(dict(result_error=result_error2, begin=i, end=i + length))
                        if len(error_list) > 0:
                            min_error = sorted(error_list, key=lambda x: x['result_error'], reverse=False)[0]
                            result_dict.append(
                                dict(error=min_error['result_error'], miss_id=miss_move_id, com_id=com['movement_id'],
                                     begin=min_error['begin'], end_index=min_error['end']))
                        count += 1
                print(count)
                with open(self.mid_pre + "knn_result" + str(index) + str(thread_id) + ".json", 'w+', encoding='utf-8') as w:
                    json.dump(result_dict, w)
            except BaseException as e:
                print(e)
                continue

    def predict_speed_for_abnormal(self, file_index):
        data = pd.read_csv(self.mid_pre + "complete" + str(file_index) + '.csv')
        miss_data = pd.read_csv(self.mid_pre + "miss_move_fill" + str(file_index) + '.csv')
        miss_data_group = miss_data.groupby('movement')
        data_group = data.groupby('movement')
        movement_id = data_group.movement.indices.keys()

        new_data = pd.DataFrame()
        for move_id in movement_id:
            move = data_group.get_group(move_id).loc[:, ['gps_v', 'movement']]
            move = move.reset_index().reset_index()
            if new_data.empty:
                new_data = move
            else:
                new_data = pd.concat([move, new_data])
        new_data.drop(['index'], axis=1, inplace=True)
        new_data.columns = ['X', "Y", "movement"]
        new_data_group = new_data.groupby("movement")

        with open(self.mid_pre + "knn_result__" + str(file_index) + ".json", 'r', encoding='utf-8') as r:
            knn_result = json.load(r)
        knn_result_df = pd.DataFrame(knn_result)
        knn_result_group = knn_result_df.groupby("miss_id")
        knn_result_id = knn_result_group.miss_id.indices.keys()
        new_miss_data = pd.DataFrame()
        print("knn movement id num:", len(knn_result_id))

        for miss_id in knn_result_id:
            try:
                miss_part_data = knn_result_group.get_group(miss_id)
                miss_part_data = miss_part_data.sort_values(by='error', ascending=True).head(20)

                miss_id_data = miss_data_group.get_group(miss_id).reset_index()
                abnormal_index = miss_id_data.loc[miss_id_data['abnormal'] == 1, :].index

                all_length = miss_id_data.shape[0]
                max_abnor_index = max(abnormal_index)
                min_abnor_index = min(abnormal_index)
                miss_len = len(abnormal_index)
                if miss_len >=  all_length* 0.5:
                    print("缺失值太长，放弃。。。")
                    continue
                print("缺失值长度:", miss_len)
                if miss_len < 10:
                    half_window = 50
                else:
                    half_window = miss_len * 2
                max_index = min(max_abnor_index + half_window, all_length)
                min_index = max(min_abnor_index - half_window, 0)
                test_data = miss_id_data.iloc[min_index:max_index, :]
                test_y = test_data.gps_v
                test_x = np.array(test_data.index)
                train_data = pd.DataFrame()
                for row in miss_part_data.itertuples():
                    try:
                        if file_index == 2:
                            com_id = row.com_id + 1673
                        elif file_index == 3:
                            com_id = row.com_id + 3072
                        else:
                            com_id = row.com_id
                        train_data_for_part = new_data_group.get_group(com_id)
                        train_data_for_part = train_data_for_part.iloc[row.begin:row.end_index, :]
                        train_data_for_part = train_data_for_part.reset_index().reset_index().loc[:, ['level_0', 'Y']]
                        train_data_for_part.columns = ['X', 'Y']
                        if not train_data.empty:
                            train_data = pd.concat([train_data_for_part, train_data])
                        else:
                            train_data = train_data_for_part
                    except BaseException as e:
                        print(e)
                        print("数据丢失")

                y_pred = train_data.groupby('X')['Y'].mean().values
                # model = linear_model.LinearRegression()
                # model = model.fit(train_data['X'].reshape(-1, 1), train_data['Y'].reshape(-1, 1))
                # y_pred = model.predict(test_x.reshape(-1, 1))
                # y_pred = self.search(train_data['X'].reshape(-1, 1), train_data['Y'].reshape(-1, 1), test_x)
                miss_id_data.loc[min_abnor_index:max_abnor_index, 'gps_v'] = y_pred[min_abnor_index: max_abnor_index + 1]
                if not new_miss_data.empty:
                    new_miss_data = pd.concat([new_miss_data, miss_id_data])
                else:
                    new_miss_data = miss_id_data
                # draw_double_line(test_x, y_pred, test_y, '预测' + str(miss_id))
            except BaseException as e:
                print(e)
                print("异常")
        print("拯救的数据样本:",file_index, new_miss_data.shape)
        print("拯救的运动学片段数量:",file_index, len(set(new_miss_data['movement'])))
        new_miss_data.to_csv(self.mid_pre + "new_miss_data" + str(file_index) + ".csv")

    def search(self, x_train, y_train, x_test):
        xgb_model = xgb.XGBModel()
        params = {
            'booster': ['gblinear'],
            'silent': [1],
            'learning_rate': [x for x in np.round(np.linspace(0.01, 1, 20), 2)],
            'reg_lambda': [lambd for lambd in np.logspace(0, 3, 50)],
            'objective': ['reg:linear']
        }
        print('begin')
        clf = GridSearchCV(xgb_model, params,
                           scoring='neg_mean_squared_error',
                           refit=True)

        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        return preds

    def cal_idling_speed(self, data):
        idling_queue = []
        data['time'] = data['time'].apply(lambda x: time.mktime(x.timetuple()))
        print("去除怠速之前：", data.shape)
        waste_queue = []
        for row in data.itertuples():
            if row.gps_v < 10:
                idling_queue.append((row.order, row.time))
                # 因为时间不一定连续，所以这里用时间来计算差异
                if len(idling_queue) > 2 and idling_queue[-1][1] - idling_queue[0][1] > 180:
                    waste_queue.append(idling_queue.pop(0)[0])
            else:
                if len(idling_queue) >= 1:
                    idling_queue.pop(0)
        useful_queue = set(data.order) - set(waste_queue)
        data = data.loc[data['order'].isin(useful_queue)]
        print("去除怠速之后：", data.shape)
        return data

    def change_data_format(self):
        self.data1 = pd.read_excel(self.raw_pre + "new1.xlsx")
        self.data2 = pd.read_excel(self.raw_pre + "new2.xlsx")
        self.data3 = pd.read_excel(self.raw_pre + "new3.xlsx")
        self.data1.to_csv(self.raw_pre + "new1.csv")
        self.data2.to_csv(self.raw_pre + "new2.csv")
        self.data3.to_csv(self.raw_pre + "new3.csv")

    def find_movement_state(self, data):
        """
        计算行驶工况
        :param data:
        :return:
        """
        flag = False

        movement_list = []
        for row in data.itertuples():
            if row.time_split:
                self.movement_state += 1
                flag = False
            else:
                if row.gps_v == 0 and flag:
                    flag = False
                    self.movement_state += 1
                elif row.gps_v != 0:
                    flag = True
            movement_list.append(deepcopy(self.movement_state))
        self.movement_state += 1
        data['movement'] = movement_list
        waste_movement = set(data.loc[data.time_split >= 2, 'movement'].values)
        waste_index = data.loc[data.movement.isin(waste_movement),:].index
        print(data.shape)
        data.drop(waste_index, inplace=True, axis=0)
        print("去除异常时间点的样本数:", data.shape)
        print("去除异常时间点的运动片段:", len(set(data.loc[data.time_split >= 2, 'movement'].values)))
        return data

    def do_cal_move_feature(self, index):
        data = pd.read_csv(self.mid_pre + "complete" + str(index) + ".csv")
        self.cal_movement_feature(data)

    def cal_final_move_feature(self):
        data = pd.read_excel(self.result_pre + "final_move_data.xlsx")
        data['time'] = data.index.values
        data = self.cal_acc_speed(data, final=True)
        self.do_cal_feature(-1, data)
        move_feature = pd.DataFrame(self.move_feature).T
        move_feature.to_csv(self.result_pre + "move_feature_final.csv")
        plot_v_t(data.index.values, data.gps_v.values, '模拟工况')

    def cal_movement_feature(self, data, final=False):
        """
        计算行驶工况的特征参数
        :return:
        """
        data = data.loc[:, ['time', 'movement', 'gps_v', 'acc']]
        data_group = data.groupby("movement")
        movement_id = data_group.movement.indices.keys()
        for id in movement_id:
            movement_state = data_group.get_group(id)
            self.do_cal_feature(id, movement_state)

        move_feature_df = pd.DataFrame(self.move_feature)
        move_feature_df = move_feature_df.apply(lambda x: np.round(x, 3))
        move_feature_df.fillna(0, inplace=True)
        if not final:
            pre = self.mid_pre
        else:
            pre = self.result_pre
        move_feature_df.to_csv(pre + "move_feature.csv", index=False)
        data.to_csv(pre + "move_step.csv", index=False)

    def do_cal_feature(self,id, movement_state):
        a_thresh = 0.1
        # 平均速度
        v_m = movement_state.gps_v.mean()
        # 平均运行速度
        v_mr = movement_state.loc[movement_state.gps_v > 0, 'gps_v'].mean()
        # 最大速度
        v_max = movement_state.gps_v.max()
        # 平均加速度
        a_m1 = movement_state.loc[movement_state.acc > a_thresh].acc.mean()
        # 平均减速度
        a_m2 = movement_state.loc[movement_state.acc < -a_thresh].acc.mean()

        a_max = movement_state.acc.max()
        a_min = movement_state.acc.min()
        # 计算样本方差
        a_std = movement_state.acc.std()
        v_std = movement_state.gps_v.std()

        # 总时间
        total_time = movement_state.shape[0]

        # 加速比例
        p_a = movement_state.loc[movement_state.acc > a_thresh].shape[0] / total_time

        # 减速比例
        p_d = movement_state.loc[movement_state.acc < -a_thresh].shape[0] / total_time

        # 怠速比例
        p_i = movement_state.loc[(movement_state.acc.abs() <= a_thresh) & (
                movement_state.gps_v <= self.idling_v_thresh)].shape[0] / total_time

        # 匀速比例
        p_c = (1 - p_a - p_d - p_i) / total_time

        self.move_feature.append(
            dict(movement=id, v_m=v_m, v_mr=v_mr, v_max=v_max, a_m1=a_m1, a_m2=a_m2, a_max=a_max, a_min=a_min,
                 v_std=v_std, a_std=a_std, p_a=p_a, p_d=p_d, p_i=p_i, p_c=p_c, total_time=total_time))

    def get_top_k_movement(self):
        move_feature = pd.read_csv(self.mid_pre + "move_feature.csv")
        move_data = pd.read_csv(self.mid_pre + "move_step.csv")

        move_feature = move_feature.sort_values(by='total_time', ascending=False)
        move_feature = move_feature.loc[:600, :]
        top_move_feature = move_feature.sample(n=10)['movement']
        move_data = move_data.loc[move_data.movement.isin(top_move_feature), ['movement', 'gps_v']]
        move_data2 = move_data.groupby('movement')
        movement_id = move_data2.movement.indices.keys()
        new_data = pd.DataFrame()
        for move_id in movement_id:
            data = move_data2.get_group(move_id)
            data = data.reset_index().reset_index()
            if new_data.empty:
                new_data = data
            else:
                new_data = pd.concat([new_data, data])
        new_data.drop(['index'], inplace=True, axis=1)
        new_data.to_excel(self.mid_pre + "top_k_move_step.xlsx", index=False)

    def do_pca(self):
        # pca 降纬
        self.pca = PCA(n_components=6)
        move_feature = pd.read_csv(self.mid_pre + "move_feature.csv")
        movement_id = move_feature['movement']
        move_feature.drop(['movement'], axis=1, inplace=True)
        columns = move_feature.columns
        move_feature_value = scale(move_feature.values)
        self.pca.fit(move_feature_value)
        tmp = pd.DataFrame(self.pca.components_)
        tmp.columns = columns
        tmp.to_csv("pca_主成分得分表.csv")
        tmp2 = pd.DataFrame(self.pca.explained_variance_ratio_)
        tmp2.to_csv("pca_explained_variance_ratio.csv")

        tmp3 = pd.DataFrame(self.pca.explained_variance_)
        tmp3.to_csv("pca_explained_ratio.csv")
        new_feature = self.pca.transform(X=move_feature_value)
        # new_feature = scale(new_feature)
        new_feature = pd.DataFrame(new_feature)
        sigma = new_feature.mean() + 2 * new_feature.std()

        shape0 = new_feature.shape[0]
        new_feature['movement'] = movement_id
        # index = {}
        # for i in range(6):
        #     temp = set((new_feature.loc[(new_feature.iloc[:,i] < sigma[i]) & (new_feature.iloc[:,i] >- sigma[i]), :].index.values))
        #     if i == 0:
        #         index = temp
        #     else:
        #         index = index & temp
        # print(len(index) / shape0)
        #
        # new_feature = new_feature.loc[new_feature.index.isin(index), :]
        # new_feature.to_csv("pca_transform.csv")
        # kmeans 聚类
        self.km = KMeans(n_clusters=2, random_state=9)
        y_pred = self.km.fit_predict(new_feature.iloc[:, [0, 1, 2, 3, 4, 5]])
        print(metrics.calinski_harabaz_score(new_feature, y_pred))
        self.showresult(new_feature.iloc[:, 0].values, new_feature.iloc[:, 1].values, new_feature.iloc[:, 1].values, y_pred)
        plt.scatter(new_feature.iloc[:, 0], new_feature.iloc[:, 1], c=y_pred)

        plt.savefig("cluster.jpg")
        new_feature['cluster'] = y_pred
        self.cal_cp_sp(new_feature)
        move_feature['movement'] = movement_id
        cluster_feature = pd.merge(move_feature, new_feature, on='movement')
        feature_cluster = cluster_feature.groupby('cluster')
        cluster_ids = feature_cluster.cluster.indices.keys()
        useful_columns = ['a_m1', 'a_m2', 'a_max', 'a_min', 'a_std', 'p_a', 'p_c', 'p_d', 'p_i', 'v_m', 'v_max', 'v_mr',
                          'v_std', 'total_time']
        avg_columns = ['total_time', 'p_a', 'p_d', 'p_i', 'p_c']
        max_columns = ['a_max', 'v_max']
        min_columns = ['a_min']

        other_columns = set(cluster_feature.columns) - set(avg_columns) - set(max_columns) - set(min_columns) - {
            'movement', 'cluster'}
        other_columns = list(other_columns)
        all_move_time = cluster_feature['total_time'].sum()

        all_corr_list = []
        move_data = pd.read_csv(self.mid_pre + "move_step.csv")
        for clu in cluster_ids:
            corr_list = []
            cluster = feature_cluster.get_group(clu)
            cluster_movement = cluster['movement'].values

            cluster_mean = cluster.loc[:, avg_columns].mean().reset_index()
            cluster_max = cluster.loc[:, max_columns].max().reset_index()
            cluster_min = cluster.loc[:, min_columns].min().reset_index()
            cluster_other = cluster.loc[:, other_columns + ['total_time']]
            cluster_total_time = cluster.loc[:, 'total_time'].sum()

            time_ratio = cluster_total_time / all_move_time
            print(clu, "运行时间占比：", time_ratio)

            other_list = []
            cluster_combine_feature = pd.concat([cluster_mean, cluster_max, cluster_min])
            cluster_combine_feature.columns = ['index', 'value']
            for col in other_columns:
                tmp = (cluster_other[col] * cluster_other['total_time']).sum() / cluster_total_time
                other_list.append(dict(index=col, value=tmp))
            other_list = pd.DataFrame(other_list)
            cluster_combine_feature = pd.concat([cluster_combine_feature, other_list])
            cluster_combine_feature.to_csv(self.result_pre + "cluster_combine_feature" + str(clu) + ".csv")
            special_columns = ['movement', 'cluster']
            cluster.drop(special_columns, axis=1, inplace=True)
            for i in range(cluster.shape[0]):
                row = cluster.iloc[i, :]
                line = pd.merge(row.reset_index(), cluster_combine_feature, on='index')
                corr = np.corrcoef(line.iloc[:, 1], line.iloc[:, 2])[0, 1]
                corr_list.append(dict(movement=cluster_movement[i], corr=corr, clu=clu, time=row.total_time))
            corr_list = sorted(corr_list, key=lambda x: x['corr'], reverse=True)
            this_move_time = 0

            for item in corr_list:
                if this_move_time < self.MOVE_TIME * time_ratio:
                    acc_list = move_data.loc[move_data.movement == item['movement'], 'acc']
                    now_total_time = sum(map(lambda x: x['time'], self.FINAL_MOVEMENT))
                    if len(list(filter(lambda x: x > 4 or x < -8, acc_list))) == 0 and (now_total_time + item['time'] < 1300):
                        this_move_time += item['time']
                        self.FINAL_MOVEMENT.append(item)
                else:
                    break
            all_corr_list += corr_list
        corr_df = pd.DataFrame(all_corr_list)
        corr_df.to_csv(self.mid_pre + "corr.csv", index=False)
        actual_total_time = sum(map(lambda x: x['time'], self.FINAL_MOVEMENT))
        final_movement_id = list(map(lambda x: x['movement'], self.FINAL_MOVEMENT))
        print(self.FINAL_MOVEMENT)

        final_move_data = move_data.loc[move_data.movement.isin(final_movement_id), :].reset_index()
        final_move_data.drop(['index'], inplace=True, axis=1)
        final_move_data.to_excel(self.result_pre + "final_move_data.xlsx")
        print("最终运行工况时长:", actual_total_time)

    def update_knn_result(self):
        with open(self.mid_pre + 'knn_result2.json', 'r') as r:
            data = json.load(r)
        res = []
        for item in data:
            res.extend(item)

        with open(self.mid_pre + 'knn_result__2.json', 'w') as w:
            json.dump(res, w)


        with open(self.mid_pre + 'knn_result3.json', 'r') as r:
            data = json.load(r)

        res = []
        for item in data:
            res.extend(item)

        with open(self.mid_pre + 'knn_result__3.json', 'w') as w:
            json.dump(res, w)

    def update_miss_move_fill(self):
        data = pd.read_csv(self.mid_pre + 'miss_move_fill.csv')
        data = data.drop(data.loc[data.abnormal == 1].index)
        data = data.apply(lambda x: np.round(scale(x),3))
        data.to_csv("miss_move_fill_scale.csv")

    def cal_cp_sp(self, data):
        data_group = data.groupby("cluster")
        clu_id = data_group.cluster.indices.keys()
        total_clu_mean = 0
        total_clu_std = 0
        for clu in clu_id:
            clu_df = data_group.get_group(clu)
            clu_df = clu_df.loc[:, [0, 1, 2, 3, 4, 5]] - self.km.cluster_centers_[clu]
            clu_df = clu_df.apply(lambda x: x ** 2)
            clu_df = clu_df.apply(lambda x: np.sqrt(x))
            clu_mean = clu_df.mean().mean()
            clu_std = clu_df.std().std()

            total_clu_mean += clu_mean
            total_clu_std += clu_std
        print("均值：", total_clu_mean / 2)
        print("方差:", total_clu_std / 2)
        sp = self.km.cluster_centers_[0] - self.km.cluster_centers_[1]
        print("sp均值：",sp.mean())
        print("方差:", sp.std())

    def showresult(self, arrX, arrY, arrZ, res):
        x1 = []
        y1 = []
        z1 = []
        x2 = []
        y2 = []
        z2 = []
        for i in range(len(arrX)):
            if res[i] == 0:
                x1.append(arrX[i])
                y1.append(arrY[i])
                z1.append(arrZ[i])
            else:
                x2.append(arrX[i])
                y2.append(arrY[i])
                z2.append(arrZ[i])
        # 绘制散点图
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x1, y1, z1, c='r')
        ax.scatter(x2, y2, z2, c='g')
        # 绘制图例
        ax.legend(loc='best')
        plt.savefig("cluster3d.jpg")
        plt.show()

    def cal_multi_move_feature(self):
        data1 = pd.read_csv(self.result_pre + "move_feature_final.csv")
        data1.columns = ['index', 'final']
        data2 = pd.read_csv(self.result_pre + "cluster_combine_feature0.csv")
        data2.drop(['Unnamed: 0'], axis=1, inplace=True)
        data2.columns = ['index', 'clu1']
        data3 = pd.read_csv(self.result_pre + "cluster_combine_feature1.csv")
        data3.drop(['Unnamed: 0'], axis=1, inplace=True)
        data3.columns = ['index', 'clu2']
        data = pd.merge(data1, data2, on='index')
        data = pd.merge(data, data3, on='index')
        data['acc1'] =  (data['final'] - data['clu1']) / data['clu1'] * 100
        data['acc2'] = (data['final'] - data['clu2']) / data['clu2'] * 100

        data.iloc[:, 1:] = data.iloc[:, 1:].apply(lambda x: np.round(x, 3))
        data.to_csv("final_compare.csv")
        pass

    def compare(self):
        data = pd.read_csv("final_compare_2.csv")
        pass

if __name__ == "__main__":
    dp = DataProcess()
    # dp.compare()
    # dp.load_data()
    # dp.update_knn_result()
    # dp.change_data_format()
    # dp.main()
    # dp.load_miss_data()
    # dp.do_cal_move_feature(1)
    # dp.get_multi_miss_data()
    # dp.predict_speed_for_abnormal(1)
    # dp.predict_speed_for_abnormal(2)
    # dp.predict_speed_for_abnormal(3)
    # dp.get_top_k_movement()
    # dp.do_pca()
    dp.cal_final_move_feature()
    # dp.update_miss_move_fill()
    dp.cal_multi_move_feature()
