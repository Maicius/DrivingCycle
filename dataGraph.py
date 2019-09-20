import xlrd                                                        #导入xlrd模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class ExcelData():
    def __init__(self,data_path,sheetname):
        self.data_path = data_path                                 # excle表格路径，需传入绝对路径
        self.sheetname = sheetname                                 # excle表格内sheet名
        self.data = xlrd.open_workbook(self.data_path)             # 打开excel表格
        self.table = self.data.sheet_by_name(self.sheetname)       # 切换到相应sheet
        self.keys = self.table.row_values(0)                       # 第一行作为key值
        self.rowNum = self.table.nrows                             # 获取表格行数
        self.colNum = self.table.ncols                             # 获取表格列数
        # print(self.rowNum)
        # print(self.colNum)

    def readExcel(self):
        if self.rowNum<2:
            print("excle内数据行数小于2")
        else:
            L = []                                                 #列表L存放取出的数据
            for i in range(1, self.rowNum):                         #从第二行（数据行）开始取数据
                sheet_data = {}                                    #定义一个字典用来存放对应数据
                for j in range(self.colNum):                       #j对应列值
                    sheet_data[self.keys[j]] = self.table.row_values(i)[j]    #把第i行第j列的值取出赋给第j列的键值，构成字典
                L.append(sheet_data)                               #一行值取完之后（一个字典），追加到L列表中
            #print(type(L))
            return L


if __name__ == '__main__':
    #这里设置Excel文件名称
    data_path = "C:\\Users\\Administrator\\Desktop\\数学建模\\2019年中国研究生数学建模竞赛赛题\\final_move_data.xlsx"                                     #文件的绝对路径
    sheetname = "Sheet1"
    get_data = ExcelData(data_path, sheetname)                       #定义get_data对象
    arr = get_data.readExcel()
    df = pd.DataFrame(arr)
    list = ['red', 'yellow', 'orange', 'green', 'blue', 'purple', 'pink', 'grey', 'fuchsia', 'khaki']
    #按照movement分组并分别作图
    #df.groupby('movement').apply(lambda x: x.plot(x=0, y='gps_v', c=random.choice(list)))
    #df.groupby('movement').plot(x=0, y='gps_v', c=random.choice(list))
    df.plot(x=0, y='gps_v', c=random.choice(list))
    plt.title('Speed By Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time(s)', fontsize=12, fontweight='bold')
    plt.ylabel('Speed(km/h)', fontsize=12, fontweight='bold')
    plt.show()


