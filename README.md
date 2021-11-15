# TVM-analyze
analyze for operators and models

## 仅克隆原始数据以及生成代码

```bash
# git clone <github:url>
git clone git@github.com:dos-lab/TVMPredictor.git
```

## 克隆完整内容:

### 准备工作：

``` bash
vim ~/.ssh/config

# 添加如下内容(可以添加多个类似的块，不冲突)
Host fish.github.com
    Hostname github.com
    PreferredAuthentications publickey
    IdentityFile <你的github的RSA私钥路径>
    User <你的github用户名>
```

### 克隆命令：

```bash

# 克隆TVMPredictor仓库
git clone git@fish.github.com:dos-lab/TVMPredictor.git

cd TVMPredictor/
git submodule init
git submodule update            # 会卡在 "clone into ..." 很久，其实正在处理

# 或者直接执行 git clone --recursive git@fish.github.com:dos-lab/TVMPredictor.git
```


### 修改了子模块内容：
1. 在子模块文件夹执行

```bash
git add .
git commit -m "...
git push origin HEAD:master
```

2. 在主仓库文件夹再次执行

```bash
git add .
git commit -m "..."
git pull git@fish.github.com:dos-lab/TVM-Analyze.git master
```