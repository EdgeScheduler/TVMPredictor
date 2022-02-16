import os
cpu_num_list = [1,2,3,4,5]
for root,dirs,files in os.walk('run/'):
    for f in files:
        #print(os.path.join(root,f))
        for cpu_num in cpu_num_list:
            os.system("python {} -c {}".format(f,cpu_num))
