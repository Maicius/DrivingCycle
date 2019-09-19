import pandas as pd
import time

class DataProcess(object):
    raw_pre = 'raw_data/'
    idling_v_thresh = 10
    idling_time_thresh = 180

    def __init__(self):
        pass

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
        begin = pd.DataFrame([{'time': v_data.loc[0,'time'] - 1, 'gps_v':0}])
        length = v_data.shape[0]
        # print("data shape", v_data.shape)
        v_data2= pd.concat([begin, v_data.loc[:length - 2,:]])
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
        data = data.loc[data.order.isin(v_data3.iloc[:,0])]
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
        data =data.loc[data['order'].isin(useful_queue)]
        print("去除怠速之后：", data.shape)
        return data

    def change_data_format(self):
        self.data1 = pd.read_excel(self.raw_pre + "1.xlsx")
        self.data2 = pd.read_excel(self.raw_pre + "2.xlsx")
        self.data3 = pd.read_excel(self.raw_pre + "3.xlsx")
        self.data1.to_csv(self.raw_pre + "1.csv")
        self.data2.to_csv(self.raw_pre + "2.csv")
        self.data3.to_csv(self.raw_pre + "3.csv")

    def main(self):
        self.load_data()
        self.data1 = self.cal_acc_speed(self.data1)
        self.data1 = self.cal_idling_speed(self.data1)
        pass


if __name__ == "__main__":
    dp = DataProcess()
    dp.main()
