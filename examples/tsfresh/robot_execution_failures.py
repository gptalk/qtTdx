from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute

# 下载、加载数据可以放在外层，不受多进程限制
download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()

# 绘图测试（可选）
# import matplotlib.pyplot as plt
# timeseries[timeseries['id'] == 3].plot(subplots=True, sharex=True, figsize=(10,10))
# plt.show()
# timeseries[timeseries['id'] == 20].plot(subplots=True, sharex=True, figsize=(10,10))
# plt.show()

# ===================== 核心：所有多进程操作放主函数内 =====================
if __name__ == '__main__':
    # 1. 提取全部时序特征
    extracted_features = extract_features(
        timeseries,
        column_id="id",
        column_sort="time"
    )

    # 填充缺失/无穷值
    impute(extracted_features)

    # 2. 手动筛选特征
    features_filtered = select_features(extracted_features, y)

    # 3. 一步到位：直接提取并筛选有效特征（封装版）
    features_filtered_direct = extract_relevant_features(
        timeseries,
        y,
        column_id='id',
        column_sort='time'
    )

    # 打印维度对比
    print("全量提取特征维度：", extracted_features.shape)
    print("手动筛选后特征维度：", features_filtered.shape)
    print("一步提取筛选特征维度：", features_filtered_direct.shape)
    print("筛选后特征数量：", features_filtered.shape[1])
    print("保留特征列表：")
    for feat in features_filtered.columns:
        print(feat)
    print("\n特征数据预览：")
    print(features_filtered.head())