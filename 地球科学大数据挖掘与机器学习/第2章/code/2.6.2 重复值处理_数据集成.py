# -*- coding: utf-8 -*-

# 代码2‑3

##自定义小数定标差标准化函数
def DecimalScaler(data):
    data=data/10**np.ceil(np.log10(data.abs().max()))
    return data
