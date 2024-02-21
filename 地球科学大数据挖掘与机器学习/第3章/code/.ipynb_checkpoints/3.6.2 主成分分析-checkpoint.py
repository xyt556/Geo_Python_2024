# -*- coding: utf-8 -*-

# 代码 3‑2

import numpy as np
random = np.loadtxt("./data/random.csv",delimiter = ",").T
# 计算协方差矩阵Covariance Matrix
cov_mat = np.cov(random)
print('协方差矩阵：\n', cov_mat)



# 代码3-3

#计算特征值和特征向量
eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
print('特征值：', eig_val_cov)
print('特征向量：\n', eig_vec_cov)



# 代码3-4

# 计算贡献率
for i in range(0, len(eig_val_cov)):
    contribution = eig_val_cov[i]/sum(eig_val_cov)
    print('第{}主成分的贡献率：{}'.format(i+1,contribution))

# 计算累计贡献率  
for i in range(1, len(eig_val_cov)):
    accumulated_contribution = sum(eig_val_cov[:i])/sum(eig_val_cov)
    print('前{}个主成分的累计贡献率： {}'.format(i, accumulated_contribution))



# 代码 3-5

import numpy as np
random = np.loadtxt("./data/random.csv",delimiter = ",").T
# 计算相关矩阵Correlation Matrix
cor_mat = np.corrcoef(random)



# 代码3-6

# 计算特征值和特征向量
eig_val_cor, eig_vec_cor = np.linalg.eig(cor_mat)
print('特征值：', eig_val_cor)
print('特征向量：\n', eig_vec_cor)



# 代码 3-7

# 计算贡献率
for i in range(0, len(eig_val_cor)):
    contribution = eig_val_cor[i]/sum(eig_val_cor)
    print('第{}主成分的贡献率：{}'.format(i+1,contribution))
# 计算累计贡献率
for i in range(1, len(eig_val_cor)):
    accumulated_contribution = sum(eig_val_cor[:i])/sum(eig_val_cor)
    print('前{}个主成分的累计贡献率： {}'.format(i, accumulated_contribution))