# encoding: utf-8

# Directed graph (each unordered pair of nodes is saved once): CA-AstroPh.txt
# Collaboration network of Arxiv Astro Physics category (there is an edge if authors coauthored at least one paper)
# Nodes: 18772 Edges: 396160
# FromNodeId	ToNodeId


import sys
import numpy as np
import snap
import random
# import numpy
from numpy import *
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

reload(sys)
sys.setdefaultencoding('utf8')

def get_list(f):
    with open(f, 'r') as data:
        while True:
            lines = data.readline()
            if not lines:
                break
                pass
            x, y = [i for i in lines.split('\t')]
            a.append(x)

            b.append(y)
            pass
        pass


def snap_network():
    # G1 = snap.TNGraph.New()
    # 合并
    c = a + b
    # 去掉相同的元素
    c = list(set(c))

    for k in range(len(c)):
        G1.AddNode(int(c[k]))

    for k in range(len(a)):
        G1.AddEdge(int(a[k]), int(b[k]))


def picture(node, centrality, title_name, x_name, y_name):
    data = pd.DataFrame({'node': node, 'centrality': centrality})
    data = data.sort_index(by="centrality", ascending=True)

    print(title_name, data[-5:])

    # y = data['centrality']
    # x = range(0, len(y))
    #
    # plt.figure(1)  # 给新图形设置一个编号，在绘制多个图形时方便
    # plt.plot(x, y, 'go', alpha=0.1)  # 绘图，‘ro’代表红色('r')的点('o')来绘图
    # plt.title(title_name)  # 设置标题
    # plt.xlabel(x_name)  # 设置x坐标
    # plt.ylabel(y_name)  # 设置y坐标
    # # 计算出百分位数(第25、50、75位)，以了解数据分布
    # perc_25 = np.percentile(y, 25)
    # perc_50 = np.percentile(y, 50)
    # perc_75 = np.percentile(y, 75)
    # # 将百分位数添加到之前生成的图形中作为参考，即百分位水平线
    # plt.axhline(perc_25, label='25th perc', c='r')
    # plt.axhline(perc_50, label='50th perc', c='y')
    # plt.axhline(perc_75, label='75th perc', c='b')
    # # 添加图例说明，loc参数让pyplot决定最佳放置位置
    # plt.legend(loc='best')
    # plt.show()
    # centrality = np.array(data["centrality"])
    # plt.title(title_name)
    # plt.xlabel(x_name)  # 设置x坐标
    # plt.ylabel(y_name)  # 设置y坐标
    # plt.boxplot(centrality, sym="co", whis=1.5)
    # plt.show()


def model_PageRank(G):
    print('**************model_PageRank*************')
    PRankH = snap.TIntFltH()
    snap.GetPageRank(G, PRankH)
    # print type(PRankH)
    index = []
    score = []
    # file_PRankH = open('./data/picture/PRankH.txt','w')
    for item in PRankH:
        # file_PRankH.write(str(item))
        # file_PRankH.write('\t')
        # file_PRankH.write(str(PRankH[item]))
        # file_PRankH.write('\n')
        index.append(item)
        score.append(PRankH[item])
    # file_PRankH.close()
    data = pd.DataFrame({'index': index, 'score': score})
    # print(len(data))
    data = data.sort_values(by=['score', 'index'])
    # 写入
    data.to_csv('./data/picture/PRankH_order.csv', sep='\t')
    print('model_PageRank = ', data[-5:])
    print("#############end file#############")
    picture(index, score, 'CloseCentr', 'index', 'score')
    # # # 可视化
    # y = data['score']
    # x = range(0, len(y))
    # #
    # plt.figure(0)
    # plt.title('PageRank')
    # plt.xlabel('index')
    # plt.ylabel('score')
    # plt.plot(x, y, alpha=0.5)
    # # 添加图例说明，loc参数让pyplot决定最佳放置位置
    # plt.legend(loc='best')
    # plt.show()
    # plt.savefig("./data/picture/PageRank_1.png")
    #
    # plt.figure(1)  # 给新图形设置一个编号，在绘制多个图形时方便
    # plt.plot(x, y, 'go', alpha=0.1)  # 绘图，‘ro’代表红色('r')的点('o')来绘图
    # plt.title('PageRank')  # 设置标题
    # plt.xlabel('index')  # 设置x坐标
    # plt.ylabel('score')  # 设置y坐标
    #
    # # 计算出百分位数(第25、50、75位)，以了解数据分布
    # perc_25 = np.percentile(y, 25)
    # perc_50 = np.percentile(y, 50)
    # perc_75 = np.percentile(y, 75)
    #
    # # 将百分位数添加到之前生成的图形中作为参考，即百分位水平线
    # plt.axhline(perc_25, label='25th perc', c='r')
    # plt.axhline(perc_50, label='50th perc', c='y')
    # plt.axhline(perc_75, label='75th perc', c='b')
    # # 添加图例说明，loc参数让pyplot决定最佳放置位置
    # plt.legend(loc='best')
    # plt.show()
    # plt.savefig("./data/picture/PageRank_2.png")
    # # 将list转化成numpy.ndarray
    # score = np.array(score)
    #
    # plt.title('PageRank')
    # plt.xlabel('score')
    # # %whis定义“须”图的长度，默认值为1.5，若whis=0则boxplot函数通过绘制sym符号图来显示盒外的所有数据值。
    # # 通过调整whis的竖直来设置异常值显示的数量
    # plt.boxplot(score, sym="co", whis=1.5)
    # plt.show()


def model_CloseCentr(G):
    print('**************Returns closeness centrality of a given node NId in Graph*************')
    node = []
    centrality = []
    for NI in G.Nodes():
        CloseCentr = snap.GetClosenessCentr(G, NI.GetId())
        node.append(NI.GetId())
        centrality.append(CloseCentr)
        # print "node: %d centrality: %f" % (NI.GetId(), CloseCentr)

    data = pd.DataFrame({'node': node, 'centrality': centrality})
    # print(len(data))
    data = data.sort_values(by=['node', 'centrality'])
    # 写入
    data.to_csv('./data/picture/CloseCentr_centrality.csv', sep='\t')
    print('CloseCentr = ', data[-5:])
    print('###############end_file###############')
    picture(node, centrality, 'CloseCentr', 'node', 'centrality')

    # # # 可视化
    # data = pd.read_csv('./data/picture/CloseCentr_centrality.csv', sep='\t')
    #
    # data = data.sort_index(by="centrality", ascending=True)
    # print(data[:3])
    #
    # y = data['centrality']
    # x = range(0, len(y))
    # plt.figure(0)
    # plt.title('ClosenessCentr')
    # plt.xlabel('node')
    # plt.ylabel('centrality')
    # plt.plot(x, y)
    # # 添加图例说明，loc参数让pyplot决定最佳放置位置
    # plt.legend(loc='best')
    # plt.show()
    # plt.savefig("./data/picture/ClosenessCentr_1.png")
    #
    # plt.figure(1)  # 给新图形设置一个编号，在绘制多个图形时方便
    # plt.plot(x, y, 'go', alpha=0.01)  # 绘图，‘ro’代表红色('r')的点('o')来绘图
    # plt.title('ClosenessCentr')  # 设置标题
    # plt.xlabel('node')  # 设置x坐标
    # plt.ylabel('centrality')  # 设置y坐标
    #
    # # 计算出百分位数(第25、50、75位)，以了解数据分布
    # perc_25 = np.percentile(y, 25)
    # perc_50 = np.percentile(y, 50)
    # perc_75 = np.percentile(y, 75)
    #
    # # 将百分位数添加到之前生成的图形中作为参考，即百分位水平线
    # plt.axhline(perc_25, label='25th perc', c='r')
    # plt.axhline(perc_50, label='50th perc', c='y')
    # plt.axhline(perc_75, label='75th perc', c='b')
    # # 添加图例说明，loc参数让pyplot决定最佳放置位置
    # plt.legend(loc='best')
    # plt.show()
    # plt.savefig("./data/picture/ClosenessCentr_2.png")
    #
    # # 将list转化成numpy.ndarray
    # centrality = np.array(data["centrality"])
    #
    # plt.title('CloseCentr')
    # plt.xlabel('centrality')
    # # %whis定义“须”图的长度，默认值为1.5，若whis=0则boxplot函数通过绘制sym符号图来显示盒外的所有数据值。
    # # 通过调整whis的竖直来设置异常值显示的数量
    # plt.boxplot(centrality, sym="co", whis=1.5)
    # plt.show()


def model_GetFarnessCentr(G):
    print('**************Returns farness centrality of a given node NId in Graph*************')
    node = []
    centrality = []
    for NI in G.Nodes():
        FarCentr = snap.GetFarnessCentr(G, NI.GetId())
        node.append(NI.GetId())
        centrality.append(FarCentr)

    data = pd.DataFrame({'node': node, 'centrality': centrality})
    data = data.sort_index(by="centrality", ascending=True)
    # 写入
    data.to_csv('./data/picture/Farness_centrality.csv', sep='\t')
    print('###############end_file###############')
    print('FarnessCentr = ', data[-5:])

    picture(node, centrality, 'FarnessCentr', 'node', 'centrality')
    # y = data['centrality']
    # x = range(0, len(y))
    # plt.figure(0)
    # plt.title('FarnessCentr')
    # plt.xlabel('node')
    # plt.ylabel('centrality')
    # plt.plot(x, y)
    # # 添加图例说明，loc参数让pyplot决定最佳放置位置
    # plt.legend(loc='best')
    # plt.show()
    # plt.savefig("./data/picture/ClosenessCentr_1.png")
    # plt.figure(1)  # 给新图形设置一个编号，在绘制多个图形时方便
    # plt.plot(x, y, 'go', alpha=0.1)  # 绘图，‘ro’代表红色('r')的点('o')来绘图
    # plt.title('FarnessCentr')  # 设置标题
    # plt.xlabel('node')  # 设置x坐标
    # plt.ylabel('centrality')  # 设置y坐标
    # # 计算出百分位数(第25、50、75位)，以了解数据分布
    # perc_25 = np.percentile(y, 25)
    # perc_50 = np.percentile(y, 50)
    # perc_75 = np.percentile(y, 75)
    # # 将百分位数添加到之前生成的图形中作为参考，即百分位水平线
    # plt.axhline(perc_25, label='25th perc', c='r')
    # plt.axhline(perc_50, label='50th perc', c='y')
    # plt.axhline(perc_75, label='75th perc', c='b')
    # # 添加图例说明，loc参数让pyplot决定最佳放置位置
    # plt.legend(loc='best')
    # plt.show()
    # plt.savefig("./data/picture/ClosenessCentr_2.png")
    # # 将list转化成numpy.ndarray
    # centrality = np.array(data["centrality"])
    # plt.title('FarnessCentr')
    # plt.xlabel('centrality')
    # # %whis定义“须”图的长度，默认值为1.5，若whis=0则boxplot函数通过绘制sym符号图来显示盒外的所有数据值。
    # # 通过调整whis的竖直来设置异常值显示的数量
    # plt.boxplot(centrality, sym="co", whis=1.5)
    # plt.show()


def model_Hits(G):
    print('*********Computes the Hubs and Authorities score of every node in Graph********')
    node = []
    node2 = []
    score_NIdHubH = []
    score_NIdAuthH = []
    NIdHubH = snap.TIntFltH()
    NIdAuthH = snap.TIntFltH()
    snap.GetHits(G, NIdHubH, NIdAuthH)

    for item in NIdHubH:
        node.append(item)
        score_NIdHubH.append(NIdHubH[item])

    for item in NIdAuthH:
        node2.append(item)
        score_NIdAuthH.append(NIdAuthH[item])

    data = pd.DataFrame({'node': node, 'score_NIdHubH': score_NIdHubH, 'node2': node2, 'score_NIdAuthH': score_NIdAuthH})
    data = data.sort_values(by="score_NIdHubH", ascending=True)

    # # 写入
    data.to_csv('./data/picture/model_Hits.csv')
    print('model_Hits = ', data[-5:])

    # print('###############end_file###############')
    # # # 可视化
    # data = pd.read_csv('./data/picture/FarnessCentr.csv', sep='\t')
    # y1 = data['score_NIdHubH']
    # y2 = data['score_NIdAuthH']
    # x = range(0, len(y1))
    # plt.figure(0)
    # plt.title('Hits')
    # plt.xlabel('node')
    # plt.ylabel('score')
    # plt.plot(x, y1, 'bo', label="score_NIdHubH", alpha=0.1)
    # plt.plot(x, y2, 'r^', label="score_NIdAuthH", alpha=0.1)
    # # 添加图例说明，loc参数让pyplot决定最佳放置位置
    # plt.legend(loc='best')
    # plt.show()
    # # plt.savefig("./data/picture/ClosenessCentr_1.png")
    #
    # # 给新图形设置一个编号，在绘制多个图形时方便
    # plt.figure(2)
    # plt.plot(x, y1, 'ro', alpha=0.1)  # 绘图，‘ro’代表红色('r')的点('o')来绘图
    # plt.plot(x, y2, 'g^', alpha=0.1)
    # plt.xlabel('node')
    # plt.ylabel('score')
    # plt.title('Hits')
    # # 计算出百分位数(第25、50、75位)，以了解数据分布
    # perc_25 = np.percentile(y1, 25)
    # perc_50 = np.percentile(y1, 50)
    # perc_75 = np.percentile(y1, 75)
    #
    # # 将百分位数添加到之前生成的图形中作为参考，即百分位水平线
    # plt.axhline(perc_25, label='25th perc', c='r')
    # plt.axhline(perc_50, label='50th perc', c='y')
    # plt.axhline(perc_75, label='75th perc', c='b')
    # # 添加图例说明，loc参数让pyplot决定最佳放置位置
    # plt.legend(loc='best')
    # plt.show()
    # # plt.savefig("./data/picture/ClosenessCentr_2.png")
    #
    # plt.figure(3)
    # # 将list转化成numpy.ndarray
    # # score_NIdHubH = np.array(data["score_NIdHubH"])
    # # score_NIdAuthH = np.array(data['score_NIdAuthH'])
    # # %whis定义“须”图的长度，默认值为1.5，若whis=0则boxplot函数通过绘制sym符号图来显示盒外的所有数据值
    # # 通过调整whis的竖直来设置异常值显示的数量
    # # plt.boxplot(score_NIdHubH, sym="go", whis=1.5)
    # # plt.boxplot(score_NIdAuthH, sym="b^", whis=1.5)
    # plt.xlabel('node')
    # plt.ylabel('score')
    # plt.title('Hits')
    # # plt.show()
    # # 2
    # labels = ["score_NIdHubH", "score_NIdAuthH"]
    # # 创建5组，每一组有1000个数
    # plt.boxplot((score_NIdHubH, score_NIdAuthH), labels=labels, sym="co")
    # plt.show()


def model_degree(G):
    x = []
    y = []
    title_name = 'degree centrality'

    for NI in G.Nodes():
        DegCentr = snap.GetDegreeCentr(G, NI.GetId())
        x.append(NI.GetId())
        y.append(DegCentr)
    picture(x, y, title_name, 'node', 'centrality')


def model_between(G):
    x = []
    y = []

    title_name = 'Betweenness Centrality'
    Nodes = snap.TIntFltH()
    Edges = snap.TIntPrFltH()
    snap.GetBetweennessCentr(G, Nodes, Edges, 1.0)
    for node in Nodes:
        x.append(node)
        y.append(Nodes[node])
    picture(x, y, title_name, 'node', 'betweenness centrality')

    data = pd.DataFrame({'x': x, 'y': y})
    data = data.sort_index(by="y", ascending=True)
    # # 写入
    data.to_csv('./data/picture/model_between.csv')


def model_eigenvector(G):
    x = []
    y = []
    title_name = 'Eigenvector Centrality'
    NIdEigenH = snap.TIntFltH()
    snap.GetEigenVectorCentr(G, NIdEigenH)
    for item in NIdEigenH:
        x.append(item)
        y.append(NIdEigenH[item])
    picture(x, y, title_name, 'node', 'eigenvector centrality')


def model_analyse(G):
    # count the number of triads in G
    # print 'snap.GetTriads=', snap.GetTriads(G)
    # # get the clustering coefficient of G
    # print 'snap.GetClustCf=', snap.GetClustCf(G)
    # # Get an approximation of graph diameter
    # print 'snap.GetBfsFullDiam=', snap.GetBfsFullDiam(G, 10)

    # GetClosenessCentr
    model_CloseCentr(G)

    # GetFarnessCentr
    model_GetFarnessCentr(G)

    # GetPageRank
    model_PageRank(G)

    # GetHits
    model_Hits(G)

    # degree centrality
    model_degree(G)

    # Computes (approximate) Node and Edge Betweenness Centrality
    model_between(G)

    # Computes eigenvector centrality
    model_eigenvector(G)

    print("Analysing work is finished")


if __name__ == '__main__':
    a = []
    b = []
    file_path = './project/CA-AstroPh.txt'
    get_list(file_path)

    # 去掉末尾的'\r\n'
    b = [i.strip('\r\n') for i in b]

    # G = nx.Graph()
    # DrawGraph()
    G1 = snap.PNGraph.New()
    # snap_network()
    # print snap.GetTriads(G1)
    # print snap.GetClustCf(G1)

    # G2 = snap.LoadEdgeList(snap.PNGraph, file_path, 0, 1)
    G2 = snap.LoadEdgeList(snap.PUNGraph, file_path, 0, 1)
    # Core5 = snap.GetKCore(G2, 5)

    # # # Node Centrality Analyse
    model_analyse(G2)

    # # IMAGE ANALYSE
    # snap.PlotSccDistr(G2, "2", "Undirected graph - scc distribution")
    # snap.PlotWccDistr(G2, "3", "Undirected graph - wcc distribution")
    # snap.PlotClustCf(G2, "4", "Undirected graph - clustering coefficient")
    # snap.PlotInDegDistr(G2, "5", "Undirected graph - in-degree Distribution")
    # snap.PlotOutDegDistr(G2, "6", "Undirected graph - out-degree Distribution")
    # snap.PlotHops(G2, "7", "Undirected graph - hops", False, 1024)
    # snap.PlotShortPathDistr(G2, "8", "Undirected graph - shortest path")


    # to csv
    data = pd.DataFrame({'Source': a, 'Target': b})
    # 写入
    data.to_csv('./data/picture/network.csv', sep=',', index=False)


