import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(input_file, output_dir):
    # 读取CSV文件（只保留前两列）
    df = pd.read_csv(input_file, header=None, usecols=[0, 1])
    df.columns = ['iid', 'uid']

    # 第一次拆分：80%训练，20%临时集（后续拆分验证+测试）
    train, temp = train_test_split(
        df,
        test_size=0.2,
        random_state=42
        # 移除stratify参数，改用随机拆分
    )

    # 第二次拆分：将临时集分为50%验证和50%测试
    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=42
    )

    # 保存结果（只保留前两列）
    train[['iid', 'uid']].to_csv(f"{output_dir}/train.csv", index=False, header=False)
    val[['iid', 'uid']].to_csv(f"{output_dir}/vali.csv", index=False, header=False)
    test[['iid', 'uid']].to_csv(f"{output_dir}/test.csv", index=False, header=False)


if __name__ == '__main__':
    # 使用示例
    split_dataset("../data/Bili_Movie/Bili_Movie_pair.csv", "../data/Bili_Movie/")

