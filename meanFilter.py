import numpy as np
import matplotlib.pyplot as plt
import xlrd
import pandas as pd


ExcelFile = xlrd.open_workbook(r'C:\\Users\\Administrator\\Desktop\\数学建模\\2019年中国研究生数学建模竞赛赛题\\原始数据\\文件3.xlsx')
# 获取目标EXCEL文件sheet名
print(ExcelFile.sheet_names())

# ------------------------------------
# 若有多个sheet，则需要指定读取目标sheet例如读取sheet2
# sheet2_name=ExcelFile.sheet_names()[1]
# ------------------------------------
# 获取sheet内容【1.根据sheet索引2.根据sheet名称】
# sheet=ExcelFile.sheet_by_index(1)
sheet = ExcelFile.sheet_by_name('原始数据3')
# 打印sheet的名称，行数，列数
#print(sheet.name, sheet.nrows, sheet.ncols)
# 获取整行或者整列的值
#rows = sheet.row_values(2)  # 第三行内容
#cols = sheet.col_values(1)  # 第二列内容
#print(rows)
# 获取单元格内容
#print(sheet.cell(1, 0).value.encode('utf-8'))


#均值滤波，inputs是数组，threshold是设置的阈值
def meanF(inputs, threshold):
    tempArr = []
    count = 0
    res = []
    for x in inputs:
        if count < threshold:
            tempArr.append(x)
            narray = np.array(tempArr)
            res.append(narray.mean())
            count += 1
        else:
            tempArr.pop(0)
            tempArr.append(x)
            narray = np.array(tempArr)
            res.append(narray.mean())
    return res

#sheet.col_values(1)可以获取Excel列内容，1——gps速度；2——x方向加速度；3——y方向加速度；4——z方向加速度
#这里对传感器的三维参数做平方和并开方
arr = []
for i in range(len(sheet.col_values(1))):
    arr.append((sheet.col_values(2)[i] ** 2 + sheet.col_values(3)[0] ** 2 + sheet.col_values(4)[0] ** 2) ** 0.5)
print(arr)
plt.plot(arr)
result = meanF(arr, 4)
print(result)
plt.plot(result)


#这里对单个传感器的值做均值滤波
#plt.plot(sheet.col_values(3))
#print(sheet.col_values(3))
#result = ArithmeticAverage(sheet.col_values(3), 2)
#result = meanF(sheet.col_values(3), 3)
#plt.plot(result)
plt.show()
