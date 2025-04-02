import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import sys

import torch

sys.path.insert(0, "./")
from utils.cluster_function import (
    DistMatricesSection,
    OdClustering,
    OdMajorClusters,
    evaluate_trajectory,
    clusterPlot,
    Silhouette,
)
import pickle
import pudb


def load_data():
    file_path = "./data/trajs.pt"
    data = torch.load(file_path)
    root = []
    paths = []
    for sample in data:
        tra = sample["root"][:60, :]
        tra = tra - tra[0, :]
        root.append(tra)
        paths.append(sample["hash"])
    print("trajectorres number: ", len(root))
    return root, paths


def vis_trajectory(trajectory):
    for tra in trajectory:
        plt.plot(tra[:, 0], tra[:, 1])
    plt.savefig("trajectories.jpg")


def vis_goal(trajectory):
    goal_x = []
    goal_y = []
    for tra in trajectory:
        goal_x.append(tra[-1, 0])
        goal_y.append(tra[-1, 1])
    plt.scatter(goal_x, goal_y)
    plt.savefig("goal.jpg")


def evaluate(trajectories):
    # file = open("./../data/mot15distMatrices.pickle", 'rb')
    file = open("./output/mot15/distances/distMatrices.pickle", "rb")
    distMatrices = pickle.load(file)

    nClusDestSet = [4]
    endLabels, endPoints, nClusEnd = OdClustering(
        funcTrajectories=trajectories,
        nClusDestSet=nClusDestSet,
        shuffle=False,
        nIter=1,
        visualize=False,
    )

    startLabels = np.ones(537)
    # startLabels = np.ones(endLabels.shape[0])
    nClusStart = 1
    refTrajIndices, odTrajLabels = OdMajorClusters(
        trajectories=trajectories,
        startLabels=startLabels,
        endLabels=endLabels,
        threshold=10,
        visualize=True,
        test=True,
        load=False,
    )

    clusRange, nIter, test = list(range(2, 30)), 3, False
    # evalMeasures, tableResults = evaluate_trajectory(clusRange=clusRange, nIter=nIter, test=test,
    #                                                  distMatrices=distMatrices,
    #                                                  trajectories=trajectories, odTrajLabels=odTrajLabels,
    #                                                  refTrajIndices=refTrajIndices, nClusStart=nClusStart,
    #                                                  nClusEnd=nClusEnd,
    #                                                  modelList=None, dataName="mot15")


from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS


def plot(trajectories, hashes):
    file_path = "./"
    file = open(f"{file_path}/output/mot15/distances/distMatrices.pickle", "rb")
    distMatrices = pickle.load(file)
    for distMatrix, f in distMatrices:
        if f == "mot15_LcssMatrix_param1":
            print(f"calculating {f}")
            model = AgglomerativeClustering(affinity="precomputed", linkage="average")
            model.n_clusters = 28 * 4 * 4
            S, closestCluster, labels, subDistMatrix, shufSubDistMatrix = Silhouette(
                model=model, distMatrix=distMatrix
            )  # , trajIndices=trajIndices)
            clusterPlot(
                trajectories,
                hashes,
                model,
                distMatrix,
                trajIndices=None,
                S=S,
                closestCluster=closestCluster,
                title=None,
                plotTrajsTogether=True,
                plotTrajsSeperate=True,
                plotSilhouette=True,
                plotSilhouetteTogether=True,
                darkTheme=False,
                file_path=file_path,
            )


def d_cluster():
    distMatrices = DistMatricesSection(trajectories=trajectories, test=False)

    # # _,_,_ = OdClustering(funcTrajectories=trajectories, shuffle=True, nIter=10,nClusDestSet=[i for i in range(1, 30)], visualize=False)

    # nClusDestSet = [4]
    # # modelNames=['KMedoids','KMeans','average Agglo-Hierarch','ward Agglo-Hierarch','BIRCH','GMM']
    # modelNames = ['average Agglo-Hierarch']
    # endLabels, endPoints, nClusEnd = OdClustering(funcTrajectories=trajectories, nClusDestSet=nClusDestSet,
    #                                               modelNames=modelNames, shuffle=False, nIter=1, visualize=True)

    # startLabels = np.ones(endLabels.shape[0])
    # refTrajIndices, odTrajLabels = OdMajorClusters(trajectories=trajectories, startLabels=startLabels,
    #                                                endLabels=endLabels, threshold=10, visualize=True, test=True,
    #                                                load=False)


import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import euclidean


def calc_angle_difference(traj1, traj2):
    """
    计算两条轨迹的角度差异
    :param traj1: 轨迹 1
    :param traj2: 轨迹 2
    :return: 角度差异
    """
    # 计算轨迹的方向向量
    vec1 = traj1[-1] - traj1[0]
    vec2 = traj2[-1] - traj2[0]
    # 计算向量的模
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    # 计算向量的点积
    dot_product = np.dot(vec1, vec2)
    # 计算角度余弦值
    cos_angle = dot_product / (norm1 * norm2)
    # 计算角度差异（弧度）
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return angle


def angle_difference(traj1, traj2):
    assert len(traj1) == len(traj2), "Trajectories must have the same length"
    diff1 = calc_angle_difference(traj1, traj2)
    diff2 = calc_angle_difference(traj1[: len(traj1) // 2], traj2[: len(traj2) // 2])
    return max(diff1, diff2)


def custom_distance(traj1, traj2, angle_threshold=np.pi / 360.0 * 5.0):
    """
    计算两条轨迹中对应点的偏移量的最大值
    :param traj1: 第一条轨迹，是一个二维 numpy 数组，每行代表一个点
    :param traj2: 第二条轨迹，是一个二维 numpy 数组，每行代表一个点
    :return: 对应点偏移量的最大值
    """
    # 计算对应点之间的欧氏距离（偏移量）
    pointwise_distances = np.linalg.norm(traj1 - traj2, axis=1)
    # 返回这些偏移量的最大值
    return np.max(pointwise_distances)
    """
    自定义距离度量函数，结合空间距离和角度差异
    :param traj1: 轨迹 1
    :param traj2: 轨迹 2
    :param angle_threshold: 角度差异阈值
    :return: 自定义距离
    """
    # 计算空间距离
    spatial_distance = euclidean(traj1[-1], traj2[-1])
    # 计算角度差异
    angle = angle_difference(traj1, traj2)
    # 如果角度差异超过阈值，返回一个较大的距离
    if angle > angle_threshold:
        return spatial_distance * 2.0
    return spatial_distance


def cluster_trajectories(trajectories, angle_threshold=np.pi / 360.0 * 5.0):
    """
    将输入的轨迹点聚类成原来数量的 1/4，并返回每个簇的代表轨迹
    :param trajectories: 轨迹列表
    :param angle_threshold: 角度差异阈值
    :return: 聚类标签和每个簇的代表轨迹
    """
    # 计算聚类的数量
    n_clusters = len(trajectories) // 4
    # 自定义距离矩阵
    distance_matrix = np.zeros((len(trajectories), len(trajectories)))
    for i in range(len(trajectories)):
        for j in range(i + 1, len(trajectories)):
            dist = custom_distance(trajectories[i], trajectories[j], angle_threshold)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    # 处理距离矩阵中的无穷大值
    max_finite_distance = np.max(distance_matrix[np.isfinite(distance_matrix)])
    large_value = max_finite_distance * 10  # 取一个足够大的有限值
    distance_matrix[np.isinf(distance_matrix)] = large_value

    # 创建 AgglomerativeClustering 模型
    model = AgglomerativeClustering(
        n_clusters=n_clusters, affinity="precomputed", linkage="complete"
    )
    # 进行聚类
    labels = model.fit_predict(distance_matrix)
    # 初始化代表轨迹列表
    representative_trajectories = []
    # 遍历每个簇
    for cluster_id in range(n_clusters):
        # 找出属于当前簇的所有轨迹的索引
        cluster_indices = np.where(labels == cluster_id)[0]
        # 提取属于当前簇的所有轨迹
        cluster_trajectories = [trajectories[i] for i in cluster_indices]
        # 计算该簇内所有轨迹的质心作为代表轨迹
        if cluster_trajectories:
            centroid = np.mean(cluster_trajectories, axis=0)
            representative_trajectories.append(centroid)
        else:
            representative_trajectories.append(None)
    return labels, representative_trajectories


if __name__ == "__main__":
    # import pudb; pu.db
    trajectories, hashes = load_data()
    labels, representative_trajectories = cluster_trajectories(trajectories)
    assert len(hashes) == len(labels) == len(trajectories)
    # 使用 zip 函数简化 hash2cluster 的创建过程
    hash2cluster = {hash_val: label for hash_val, label in zip(hashes, labels)}

    file_path_1 = "data/trajectoryAfterCluster.pickle"
    with open(file_path_1, "wb") as pickle_out:
        print(f"len(clusterTra): {len(representative_trajectories)}")
        pickle.dump(representative_trajectories, pickle_out)

    file_path_2 = "data/sampleHash2Cluster.pickle"
    with open(file_path_2, "wb") as pickle_out:
        print(f"len(sampleHash2Cluster): {len(hash2cluster)}")
        pickle.dump(hash2cluster, pickle_out)
