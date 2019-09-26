import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc', size=14)
image_pre = "image/"
result_data = 'result_data/'

def plot_v_t(x, y, name):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y)
    ax1.set_title("福州-莆田地区汽车行驶模拟工况", fontproperties=font)
    ax1.set_xlabel('时间 (s)', fontproperties=font)
    ax1.set_ylabel('速度 (m/s)', fontproperties=font)
    plt.savefig(result_data + name + '.jpg')
    plt.show()

def plot_v_t2(x, y, name):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(x, y)
    ax1.set_title("福州-莆田地区汽车行驶模拟工况", fontproperties=font)
    ax1.set_xlabel('时间 (s)', fontproperties=font)
    ax1.set_ylabel('速度 (m/s)', fontproperties=font)
    plt.savefig(result_data + name + '.jpg')
    plt.show()

def draw_double_line(x, y1, y2, name):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    lns1 =  ax1.plot(x, y1, color='green', label="KNN predict v")
    # ax1.plot(x, y2, color='red', label="real v")
    ax1.set_ylabel('KNN预测速度 (m/s)', fontproperties=font)
    ax1.set_title("基于滑动窗口的KNN预测速度与真实采集的速度对比图", fontproperties=font)
    ax1.set_xlabel('时间 (s)', fontproperties=font)
    ax2 = ax1.twinx()
    lns2 = ax2.plot(x, y2, 'r', label="real v")
    ax2.set_ylabel('真实采集的速度 (m/s)', fontproperties=font)

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)
    plt.savefig(image_pre + name + ".jpg")
        # plt.show()
    pass
