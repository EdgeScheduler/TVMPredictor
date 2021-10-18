import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
import sys

base_dir=os.path.dirname(__file__)
sys.path.append(base_dir)
print(base_dir)

datas = {}
with open(os.path.join(base_dir,"../op_statistics.json"),"r") as f:
    datas = json.load(f)

op_list = list(datas["origin_data"].keys())
min_list = [value["min"] for key,value in datas["origin_data"].items()]
min_to_avg_list = [value["avg"] for key,value in datas["origin_data"].items()]
avg_to_max_list= [value["max"] for key,value in datas["origin_data"].items()]

plt.figure(figsize=(40,20))
plt.bar(range(len(op_list)), avg_to_max_list, label='avg ~ max',fc = 'brown')
plt.bar(range(len(op_list)), min_to_avg_list, label='min ~ avg',tick_label = op_list,fc = 'orange')
plt.bar(range(len(op_list)), min_list, tick_label = op_list,fc = 'lightgrey', hatch='/')
plt.plot(op_list,[0.01 for x in op_list],label='Dividing line',color='green',linewidth=1,linestyle='--')

plt.legend()
plt.xlabel('TVM-OP name')
plt.ylabel('run-time proportion')
plt.title('The distribution of run-time for tvm-op in DL-models')
plt.savefig(os.path.join(base_dir,"../cartogram.png"))


# data = xlrd.open_workbook('ccc.xlsx')  #打开数据
# table = data.sheet_by_index(0)     #获取sheet1的数据
# nrows = table.nrows           #获取sheet1中的行
# plot_list = ['plot1']
# plot1_ra_list = [[0] for i in range(17)]#构建一个17X1的列表
# i = 0
# for row in range(1,nrows):       #循环读取表内数据
#   if table.cell(row,0).value == 2015.0 and table.cell(row,1).value == 'plot1':
#     print(table.cell(row, 3).value) #第三列是各个元素所占的比例
#     print(i)
#     plot1_ra_list[i][0]=(float(table.cell(row, 3).value))
#     i+=1
# #颜色列表
# color = ['y','r','snow','b','k','g','orange','c','bisque','brown','lime','aqua','coral','darkcyan','gold','teal','pink',]
# plt.figure(figsize=(8,6))
# for i in range(17):
#   plt.bar(range(len(plot1_ra_list[i])), plot1_ra_list[i],bottom=np.sum(plot1_ra_list[:i],axis = 0),label=str(i+1),tick_label = plot_list,fc = color[i])
# plt.legend()
# plt.show()