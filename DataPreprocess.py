import pandas as pd
import time
import numpy as np

class DataProcess(object):
    raw_pre = 'raw_data/'
    mid_pre = 'mid_data/'
    idling_v_thresh = 10
    idling_time_thresh = 180

    def __init__(self):
        pass

    def main(self):
        self.load_data()
        self.data1 = self.cal_acc_speed(self.data1)
        self.data1 = self.cal_idling_speed(self.data1)
        self.data1 = self.find_movement_state(self.data1)
        self.cal_movement_feature(self.data1)

    def load_data(self):

        self.data1 = pd.read_csv(self.raw_pre + "1.csv")
        self.data1.columns = ['order', 'time', 'gps_v', 'x', 'y', 'z', 'longitude', 'latitude', 'engine',
                              'torque', 'fuel', 'accelerator', 'air', 'engine_load', 'flow']
        # self.data2 = pd.read_csv(self.raw_pre + "2.csv")
        # self.data3 = pd.read_csv(self.raw_pre + "3.csv")
        self.data1['time'] = self.data1['time'].apply(lambda x: pd.to_datetime(x[:-4]))
        # self.data2['时间'] = self.data2['时间'].apply(lambda x: pd.to_datetime(x[:-4]))
        # self.data3['时间'] = self.data3['时间'].apply(lambda x: pd.to_datetime(x[:-4]))

    def cal_acc_speed(self, data):
        v_data = data.loc[:, ['time', 'gps_v']]
        print("原始数据大小：", data.shape)
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
        v_data3 = pd.concat([v_data, v_data2], axis=1)

        # 计算加速度
        v_data3['time_step'] = (v_data3['time'] - v_data3['time2'])
        v_data3 = v_data3.loc[v_data3['time_step'] <= 1, :]
        print("去除不连续值:", v_data3.shape)
        v_data3['acc'] = (v_data3['gps_v'] - v_data3['gps_v2']) / v_data3['time_step']
        max_acc = 100 / 7
        min_acc = -8
        # print(v_data3.shape)
        print("去除异常加速度之前:", data.shape)
        v_data3 = v_data3.loc[(v_data3['acc'] < max_acc) & (v_data3['acc'] > min_acc)]
        data = data.loc[data.order.isin(v_data3.iloc[:, 0])]
        data['acc'] = v_data3['acc']
        print("去除异常加速度之后：", v_data3.shape)
        return data

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
        self.data1 = pd.read_excel(self.raw_pre + "1.xlsx")
        self.data2 = pd.read_excel(self.raw_pre + "2.xlsx")
        self.data3 = pd.read_excel(self.raw_pre + "3.xlsx")
        self.data1.to_csv(self.raw_pre + "1.csv")
        self.data2.to_csv(self.raw_pre + "2.csv")
        self.data3.to_csv(self.raw_pre + "3.csv")

    def find_movement_state(self, data):
        """
        计算行驶工况
        :param data:
        :return:
        """
        index = 0
        flag = False
        movement_state = 1
        movement_list = []
        for row in data.itertuples():
            if row.gps_v == 0 and flag:
                flag = False
                movement_state += 1
            elif row.gps_v != 0:
                flag = True
            movement_list.append(movement_state)
            index += 1
        data['movement'] = movement_list
        return data

    def cal_movement_feature(self, data):
        """
        计算行驶工况的特征参数
        :return:
        """
        move_feature = []
        data = data.loc[:, ['time', 'movement', 'gps_v', 'acc']]
        data_group = data.groupby("movement")
        movement_id = data_group.movement.indices.keys()
        a_thresh = 0.5

        for id in movement_id:
            movement_state = data_group.get_group(id)
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

            # 加速时间比
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

            move_feature.append(
                dict(movement=id, v_m=v_m, v_mr=v_mr, v_max=v_max, a_m1=a_m1, a_m2=a_m2, a_max=a_max, a_min=a_min,
                     v_std=v_std, a_std=a_std, p_a=p_a, p_d=p_d, p_i=p_i, p_c=p_c, total_time=total_time))


        move_feature_df = pd.DataFrame(move_feature)
        move_feature_df = move_feature_df.apply(lambda x: np.round(x, 3))
        move_feature_df.fillna(0, inplace=True)

        move_feature_df.to_csv(self.mid_pre + "move_feature.csv", index=False)
        data.to_csv(self.mid_pre + "move_step.csv", index=False)

    def get_top_k_movement(self):
        move_feature = pd.read_csv(self.mid_pre + "move_feature.csv")
        move_data = pd.read_csv(self.mid_pre + "move_step.csv")

        move_feature = move_feature.sort_values(by='total_time', ascending=False)
        top_move_feature = move_feature.head(10)['movement']
        move_data = move_data.loc[move_data.movement.isin(top_move_feature), ['movement', 'gps_v']]

        move_data.to_excel(self.mid_pre + "top_k_move_step.xlsx")

if __name__ == "__main__":
    dp = DataProcess()
    dp.get_top_k_movement()
