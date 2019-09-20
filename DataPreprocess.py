import pandas as pd
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics, cross_validation, ensemble
from sklearn.preprocessing import scale, minmax_scale

from sklearn.linear_model import Ridge

class DataProcess(object):
    raw_pre = 'raw_data/'
    mid_pre = 'mid_data/'
    idling_v_thresh = 10
    idling_time_thresh = 180
    MOVE_TIME = 900
    FINAL_MOVEMENT = []
    miss_index = []
    def __init__(self):
        pass

    def main(self):
        self.load_data()
        self.data1 = self.cal_acc_speed(self.data1)
        self.data1 = self.cal_idling_speed(self.data1)
        self.data1 = self.find_movement_state(self.data1)
        self.get_complete_true_movement(self.data1)
        # self.cal_movement_feature(self.data1)

    def get_complete_true_movement(self, data):
        miss_movement = data.loc[data.order.isin(self.miss_index), 'movement']
        print(len(miss_movement))
        miss_data = data.loc[data.movement.isin(miss_movement)]
        miss_data.to_csv(self.mid_pre + "miss_data.csv")
        data.drop(data.loc[data.movement.isin(miss_movement)].index, inplace=True)
        print("完整的运动片段:", data.shape)
        data.to_csv(self.mid_pre + "complete.csv")

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
        v_data2.drop(['index'], axis=1, inplace=True)
        v_data3 = pd.concat([v_data, v_data2], axis=1)


        v_data3['time_step'] = (v_data3['time'] - v_data3['time2'])
        time_split_index = v_data3.loc[v_data3['time_step'] >= 180, 'index']
        self.miss_index = v_data3.loc[(v_data3['time_step'] > 1) & (v_data3['time_step'] < 180), 'index']
        v_data3['time_split'] = 0
        v_data3.loc[v_data3.index.isin(time_split_index), 'time_split'] = 1

        # v_data3 = v_data3.loc[v_data3['time_step'] <= 1, :]


        print("去除不连续值:", v_data3.shape)
        # 计算加速度
        v_data3['acc'] = (v_data3['gps_v'] - v_data3['gps_v2']) / (v_data3['time_step'] * 3.6)
        max_acc = 100 / 7
        min_acc = -8
        # print(v_data3.shape)
        print("去除异常加速度之前:", data.shape)
        v_data3 = v_data3.loc[(v_data3['acc'] < max_acc) & (v_data3['acc'] > min_acc)]
        self.miss_index += v_data3.iloc[:, 0]
        # data = data.loc[data.order.isin(v_data3.iloc[:, 0])]
        # data['acc'] = v_data3['acc']
        # print("去除异常加速度之后：", v_data3.shape)
        data['time_split'] = v_data3['time_split']
        # print(self.miss_index)
        return data

    def predict_speed_for_abnormal(self):
        data = pd.read_csv(self.mid_pre + "complete.csv")
        data_group = data.groupby('movement')
        movement_id = data_group.movement.indices.keys()
        new_data = pd.DataFrame()
        for move_id in movement_id:
            move = data_group.get_group(move_id).loc[:, ['gps_v']]
            move = move.reset_index().reset_index()
            if new_data.empty:
                new_data = move
            else:
                new_data = pd.concat([move, new_data])
        new_data.drop(['index'], axis=1, inplace=True)
        new_data.columns = ['X', "Y"]
        best_alpha = self.find_min_alpha(new_data['X'].reshape(-1, 1), new_data['Y'].reshape(-1, 1))
        model = Ridge(alpha=best_alpha)
        self.clf = ensemble.BaggingRegressor(model, n_jobs=1, n_estimators=900)
        print("交叉验证...")
        scores = cross_validation.cross_val_score(model, new_data['X'].reshape(-1, 1), new_data['Y'].reshape(-1, 1), cv=10, scoring='neg_mean_squared_error')
        print(scores)
        print("mean:" + str(scores.mean()))

        pass


    # 找到Ridge最佳的正则值
    def find_min_alpha(self, x_train, y_train):
        alphas = np.logspace(-2, 3, 200)
        # print(alphas)
        test_scores = []
        alpha_score = []
        for alpha in alphas:
            clf = Ridge(alpha)
            test_score = -cross_validation.cross_val_score(clf, x_train, y_train, cv=10,
                                                           scoring='neg_mean_squared_error')
            test_scores.append(np.mean(test_score))
            alpha_score.append([alpha, np.mean(test_score)])
        print("final test score:")
        print(test_scores)
        print(alpha_score)

        sorted_alpha = sorted(alpha_score, key=lambda x: x[1], reverse=False)
        print(sorted_alpha)
        alpha = sorted_alpha[0][0]
        print("best alpha:" + str(alpha))
        return alpha

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
            if row.time_split:
                movement_state += 1
                flag = False
            else:
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
        self.pca = PCA(n_components=2)
        move_feature = pd.read_csv(self.mid_pre + "move_feature.csv")
        movement_id = move_feature['movement']
        move_feature.drop(['movement'], axis=1, inplace=True)
        self.pca.fit(move_feature)
        new_feature = self.pca.transform(X=move_feature)

        new_feature = scale(new_feature)

        new_feature = pd.DataFrame(new_feature)
        new_feature['movement'] = movement_id

        # kmeans 聚类
        self.km = KMeans(n_clusters=3, random_state=9)
        y_pred = self.km.fit_predict(new_feature.iloc[:, [0, 1]])
        print(metrics.calinski_harabaz_score(new_feature, y_pred))
        plt.scatter(new_feature.iloc[:, 0], new_feature.iloc[:, 1], c=y_pred)
        plt.savefig("cluster.jpg")
        new_feature['cluster'] = y_pred

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
        all_move_time = move_feature['total_time'].sum()

        all_corr_list = []
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
            print(clu,"运行时间占比：", time_ratio)

            other_list = []
            cluster_combine_feature = pd.concat([cluster_mean, cluster_max, cluster_min])
            cluster_combine_feature.columns = ['index', 'value']
            for col in other_columns:
                tmp = (cluster_other[col] * cluster_other['total_time']).sum() / cluster_total_time
                other_list.append(dict(index=col, value=tmp))
            other_list = pd.DataFrame(other_list)
            cluster_combine_feature = pd.concat([cluster_combine_feature, other_list])
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
                    this_move_time += item['time']
                    self.FINAL_MOVEMENT.append(item)
                else:
                    break
            all_corr_list += corr_list
        corr_df = pd.DataFrame(all_corr_list)
        corr_df.to_csv(self.mid_pre + "corr.csv", index=False)
        actual_total_time = sum(map(lambda x:x['time'], self.FINAL_MOVEMENT))
        final_movement_id = list(map(lambda x: x['movement'], self.FINAL_MOVEMENT))
        print(self.FINAL_MOVEMENT)
        move_data = pd.read_csv(self.mid_pre + "move_step.csv")
        final_move_data = move_data.loc[move_data.movement.isin(final_movement_id), :].reset_index()
        final_move_data.drop(['index'], inplace=True, axis=1)
        final_move_data.to_excel(self.mid_pre + "final_move_data.xlsx")
        print("最终运行工况时长:", actual_total_time)


if __name__ == "__main__":
    dp = DataProcess()
    dp.main()
    # dp.predict_speed_for_abnormal()
    # dp.get_top_k_movement()
    # dp.do_pca()
