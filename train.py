import json
import random

import numpy as np

import preprocess
from model.r_net import RNet

modOpts = json.load(open('model/config.json', 'r'))['rnet']['train']

"""加载数据
print('加载数据...')
dp = preprocess.read_data('train', modOpts) # 初始化
num_batches = int(np.floor(dp.num_samples / modOpts['batch_size'])) - 1 # 计算 batch 数量

preprocess.show_sample(dp)  # 显示数据样本
# """

# """构建模型
print("构建模型...")
rnet=RNet()
rnet.build_model()
# """





