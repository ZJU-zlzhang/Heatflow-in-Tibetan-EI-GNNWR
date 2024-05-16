import pandas as pd
import os

# 指定CSV文件的路径
csv_file_path = '../Data/Grid_05_Plate_Tibet.csv'  # 请替换为你的CSV文件路径
save_dir = '../Data/Features'

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 指定需要组合的列名（除了Lon和Lat之外的其他列）
columns_to_combine = ['Moho', 'LAB', 'Topo', 'Sus', 'Tectonics', 'MeanCurv', 'Bz5', 'Ridge', 'Trench', 'Transform',
                      'YoungRift', 'Volcanos']

# 遍历每个指定的列名
for col in columns_to_combine:
    # 选取Lon, Lat和当前列的数据
    selected_data = df[['Lon', 'Lat', col]]

    # 生成目标txt文件的完整路径
    txt_file_path = os.path.join(save_dir, f'{col}.txt')

    # 保存选取的数据到txt文件，每列之间使用空格分隔，每行使用回车符分隔
    selected_data.to_csv(txt_file_path, sep=' ', index=False, header=False, line_terminator='\n')

    print(f'文件已成功保存为：{txt_file_path}')