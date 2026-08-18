# FFOpt-LAMMPS 完整中文使用手册

本手册面向第一次开发分子力场参数的用户。目标是：用户只准备
LAMMPS data 文件、实验目标和初始参数范围，编辑一个 `ffopt.in`，即可完成
BO、局域采样、ANN、主动学习和最终 LAMMPS 验证，并在中断后自动续算。

> 当前版本为 alpha。正式支持范围是分子晶体、孤立分子和分子吸附模型。
> BTAH 是软件回归体系。单质、合金、反应力场和多晶型迁移尚未作为当前版本
> 的通用能力承诺。
> 当前吸附后端把一个指定的零电荷金属 type 视为固定基底，只优化分子 types；
> 带电、多组分或也需要优化参数的基底不属于 schema 1 的支持范围。

## 1. 先理解三个文件层次

一个项目只需要保留下面的结构：

```text
my_project/
|-- ffopt.in                 # 唯一需要长期修改的科学输入
|-- data/
|   |-- bulk/                # 分子晶体体相 data
|   |-- molecule/            # 单分子 data
|   `-- adsorption/          # complex/slab/molecule 三个吸附 data
`-- runs/                    # FFOpt 自动生成，用户不要手工修改
```

机器路径和资源不写进项目，而是只配置一次：

```text
~/.config/ffopt/machines.toml
```

因此，同一个 `ffopt.in` 可以在本机、单节点和双节点之间切换；只有速度和并发
不同，参数范围、候选数量、随机种子和目标函数不随机器改变。

## 2. 安装环境

### 2.1 推荐版本

- Python 3.11
- LAMMPS 22 Jul 2025 或经过本项目测试的兼容版本
- OpenMPI 或集群提供的兼容 MPI
- PyTorch CPU 或 CUDA 版本
- Linux/SLURM 为推荐生产环境；Windows 可用于本地开发和 GPU 训练

不要安装到 Conda `base`。独立环境便于升级、回退和删除。

### 2.2 Linux 或集群安装

当前 alpha 版通过 GitHub Release wheel 分发，尚未发布到 PyPI；因此请使用下面
带版本号的完整 URL，不要把它简写成未经确认的 `pip install ffopt-lammps`。

```bash
conda create -n ffopt python=3.11 -y
conda activate ffopt
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate ffopt
conda install -c conda-forge "lammps=*=*openmpi*" openmpi -y
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m pip install \
  "ffopt-lammps[full] @ https://github.com/Leduo-Pei/ffopt-lammps/releases/download/v0.3.0a3/ffopt_lammps-0.3.0a3-py3-none-any.whl"
```

上面的命令有意安装 CPU 版 PyTorch。GPU 工作站应先按 PyTorch 官方安装选择器
安装与本机 CUDA 驱动匹配的版本，再安装 FFOpt；`[full]` 会复用已有 PyTorch。

检查安装来源：

```bash
which python
which ffopt
which lmp
which mpirun
ffopt --version
python -c "import site; print(site.ENABLE_USER_SITE)"
python -c "import ffopt, torch; print(ffopt.__version__); print(torch.__version__, torch.cuda.is_available())"
lmp -help | head
```

`site.ENABLE_USER_SITE` 应为 `False`，这样旧的 `~/.local` Python 包不会混入
新环境。若 `python -m pip check` 报缺包，应在当前 `ffopt` 环境补装，不能依赖
其他环境中的同名包。`ffopt doctor` 发现 user-site 开启时也会给出警告。FFOpt
生成的 SLURM 脚本还会自动设置 `PYTHONNOUSERSITE=1`，避免计算节点加载用户级包。

CPU 环境中 `torch.cuda.is_available()` 显示 `False` 是正常现象，不是报错。
这表示 NN/AL 使用 CPU。只有安装 CUDA 版 PyTorch 且能识别 GPU 时才会显示
`True`。

若从 GitHub Release 页面手工下载 wheel 或源码包，可用同一页面的
`SHA256SUMS.txt` 与 `sha256sum -c SHA256SUMS.txt --ignore-missing` 检查文件完整性。

### 2.3 Windows 本地安装

在 Anaconda Prompt 或 PowerShell 中：

```powershell
conda create -n ffopt python=3.11 -y
conda activate ffopt
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate ffopt
python -m pip install "ffopt-lammps[full] @ https://github.com/Leduo-Pei/ffopt-lammps/releases/download/v0.3.0a3/ffopt_lammps-0.3.0a3-py3-none-any.whl"
```

LAMMPS 和 MPI 可以由用户单独安装，随后在机器配置中填写绝对路径。路径含空格
没有问题，命令行中应使用引号。

### 2.4 开发者安装

```bash
git clone https://github.com/Leduo-Pei/ffopt-lammps.git
cd ffopt-lammps
python -m pip install -e ".[full,dev]"
python -m pytest -q
```

普通用户优先安装固定 release/tag；开发者才使用 `-e`。

不带 extra 的基础包可执行 data 检查、输入编译和 validate-only LAMMPS
计算。完整 `bo sample nn al audit finalize validate` 必须安装 `[full]`；`[saasbo]` 和
`[xgboost]` 各自包含其完整运行依赖，不要求用户先猜测组合多个 extra。

### 2.5 升级、重装和卸载

普通升级应安装一个明确的新 release，而不是跟随随时变化的 `main`。升级前先让
正在使用该环境的作业结束，并记录 `ffopt --version`。最干净的重装方法是删除并
重建独立 Conda 环境：

```bash
conda deactivate
conda env remove -n ffopt -y
conda create -n ffopt python=3.11 -y
conda activate ffopt
# 随后按 2.2 节重新安装 LAMMPS、MPI、PyTorch 和 FFOpt。
```

删除 Conda 环境不会删除用户项目、`runs/` 或
`~/.config/ffopt/machines.toml`。若要重新探测机器，先备份配置再重建：

```bash
cp -a "$HOME/.config/ffopt" "$HOME/.config/ffopt.backup"
```

确认新环境中的 `which python`、`which ffopt`、`which lmp` 和 `which mpirun`
都指向同一个环境后，再运行 `machine test` 和 `self-test`。不要在仍有 SLURM 作业
运行时删除它正在使用的环境。

## 3. 配置机器

### 3.1 自动探测

```bash
ffopt machine probe
ffopt machine probe --partition CPU
```

该命令只读取环境，不修改配置。它会报告 Python、LAMMPS、MPI、CPU、GPU 和
SLURM 分区，并给出保守建议。建议仍需结合本集群的节点共享和内存规则审核。

只有 `local` 有内置零配置 profile；集群必须先执行 `machine configure` 并使用
明确名称，不能依赖隐藏的通用 `cluster`。profile 名必须以英文字母开头，后面只
能使用英文字母、数字、点、下划线或连字符，例如 `mag1-2node`。

### 3.2 本地机器

若 `lmp` 已经在当前环境的 `PATH` 中，可以不写任何机器配置，直接使用内置
`local`：

```bash
ffopt doctor ffopt.in --machine local
ffopt run ffopt.in --machine local
```

这个零配置模式为了稳妥只启用一个串行 worker。需要指定 LAMMPS/MPI 绝对路径或
充分利用本机多核时，再创建下面的命名 profile：

```bash
ffopt machine list
ffopt machine show --name local
ffopt machine test --name local
```

列表会把它标记为 `[built-in, serial]`；如果用户显式配置同名 `local`，则显示
`[user override]`。

```bash
ffopt machine configure \
  --name local-workstation \
  --backend local \
  --lammps /absolute/path/to/lmp \
  --mpi /absolute/path/to/mpirun \
  --workers 4 \
  --mpi-ranks 4 \
  --omp-threads 1 \
  --timeout 21600 \
  --force
```

### 3.3 SLURM 单节点

以下示例假设每节点 48 核：

```bash
ffopt machine configure \
  --name cluster-1node \
  --backend slurm \
  --lammps /absolute/path/to/lmp \
  --mpi /absolute/path/to/mpirun \
  --partition CPU \
  --nodes 1 \
  --total-cores 48 \
  --workers 12 \
  --mpi-ranks 4 \
  --omp-threads 1 \
  --memory-per-node 64G \
  --walltime 14-00:00:00 \
  --timeout 216000 \
  --force
```

### 3.4 SLURM 双节点

```bash
ffopt machine configure \
  --name cluster-2node \
  --backend slurm \
  --lammps /absolute/path/to/lmp \
  --mpi /absolute/path/to/mpirun \
  --partition CPU \
  --nodes 2 \
  --total-cores 96 \
  --workers 24 \
  --mpi-ranks 4 \
  --omp-threads 1 \
  --memory-per-node 64G \
  --walltime 14-00:00:00 \
  --timeout 216000 \
  --force
```

四个 CPU 参数的关系是：

```text
最低总核数 = workers * mpi-ranks * omp-threads
```

- `workers`：同时运行多少个彼此独立的参数点。
- `mpi-ranks`：每一个参数点内部用多少个 MPI 进程跑 LAMMPS。
- `omp-threads`：每个 MPI 进程使用多少线程，通常设为 1。
- `total-cores`：向 SLURM 请求的总 CPU 数。

`workers` 只影响速度，不控制 BO 每轮候选数。BO 的科学预算由 `ffopt.in` 中
的 `batch_size` 决定。

`walltime` 应按任务真实上限填写。过长的时限可能让共享集群无法回填当前空闲核，
导致小型验收任务看似有资源却长期显示 `Priority`；软件验收通常用 4--6 小时，
正式长模拟再使用 14 天等生产时限。

单个 LAMMPS 参数点的 `timeout` 必须短于 `walltime`。这样某个病态参数异常缓慢时，
FFOpt 可以及时判失败并继续其他候选，还能留下时间写 checkpoint；6 小时验收任务
建议从 `--timeout 7200`（2 小时）开始。

多次执行 `machine configure` 时，不加 `--force` 会拒绝覆盖；加 `--force`
只替换同名 profile，不会追加重复表，也不会删除其他机器配置。

### 3.5 机器验收

```bash
ffopt machine list
ffopt machine show --name cluster-1node
ffopt machine test --name cluster-1node
ffopt self-test --machine cluster-1node --watch
```

`machine test` 使用极小 LAMMPS/MPI 作业，通常几分钟内完成。单节点 profile 测试
一个执行 slot；多节点 profile 会按其真实 `nodes`、`workers` 和 `mpi-ranks` 启动
分布式 worker pool，在所有 slot 上并发执行并确认实际覆盖全部节点，因此可以在
科学计算前发现跨节点共享文件系统、MPI 或资源映射问题。`self-test`
会执行真实的 BTAH NPT、BO、采样、ANN、AL、审计和最终验证，通常需要数小时；
它包括只在末端执行的吸附模块，并检查最终 objective、性质误差和 tolerance。
新服务器正式计算前必须先通过两者，不能因为 `self-test` 没有几分钟结束就判断
程序卡住，应使用 `squeue`、`ffopt status` 和 `ffopt logs` 查看进度。

## 4. 准备 LAMMPS data 文件

### 4.1 支持的来源

data 文件可以来自 Materials Studio 的 CAR/MDF 经 `msi2lmp` 转换，也可以由
其他工具生成。FFOpt 不依赖 MS，但要求文件符合 LAMMPS data 格式，并包含与
所用输入模板一致的拓扑、Masses、Pair Coeffs 和 Atoms 信息。

Windows 文本中的 `^M` 是 CRLF 换行在某些编辑器中的显示。LAMMPS 通常能够
读取；FFOpt 也以通用换行方式解析。它不等于文件内容损坏。若需要统一格式，可
用 `dos2unix file.data`，但不要在计算途中随意改动源文件。

### 4.2 一个 type 一个参数集合

FFOpt 当前假设同一个 type 的所有原子共享同一组 epsilon、sigma 和 q。如果
同一个 type 在 Atoms 段出现多个不同电荷，必须先拆分 type，否则无法用一行
`type` 参数准确表示。

### 4.3 文件角色

- `bulk`：完整周期性分子晶体超胞。
- `single`：与 bulk 中同一分子、同一 type 编号和拓扑的孤立分子。
- `complex`：吸附分子和基底的组合体系。
- `slab`：与 complex 完全一致的干净基底。
- `molecule`：与 complex 完全一致的孤立吸附分子。

推荐命名：

```text
BTAH_bulk.data
BTAH_single.data
BTAH_Au111_complex.data
BTAH_Au111_slab.data
BTAH_Au111_molecule.data
```

不要使用 `new.data`、`final2.data`、`test-ok.data` 等无法追溯角色的名称。

`type` 标签必须以英文字母开头，后面只能使用英文字母、数字和下划线，例如
`bhN1`、`C_aromatic_1`。不要使用空格、斜杠、连字符或中文；FFOpt 会用标签构造
参数名和 CSV 列名，并在运行前拒绝不安全的手写标签。

### 4.4 检查 data

```bash
ffopt inspect BTAH_bulk.data
ffopt data check --bulk BTAH_bulk.data --single BTAH_single.data --strict
```

吸附体系：

```bash
ffopt data check \
  --complex BTAH_Au111_complex.data \
  --slab BTAH_Au111_slab.data \
  --molecule BTAH_Au111_molecule.data \
  --strict
```

`--strict` 把 warning 也视为失败，适合生产前验收。

### 4.5 命名规则速查

| 对象 | 推荐形式 | 限制或含义 |
|---|---|---|
| `project` | `benzene_charge_only` | 英文字母开头；只用字母、数字、`.`、`_`、`-` |
| machine profile | `ccelab-2node` | 与 project 相同的可移植字符规则 |
| type label | `C_aromatic_1` | 英文字母开头；只用字母、数字、`_` |
| data 文件 | `benzene_bulk.data` | 写清材料、角色，必要时再加晶面或构型 |
| run ID | `production_01` | 只用字母、数字、`.`、`_`、`-` |
| 内部参数列 | `C_aromatic_1_charge` | 软件由 `type label + 参数族` 自动生成，不手写 |

路径可以包含空格或 UTF-8 字符；命令行中的含空格路径必须加引号，`ffopt init`
写入 `ffopt.in` 时会自动为需要的路径加引号。名称中不要写机器核数、日期或
“final”等会很快失真的状态；节点资源属于 machine profile，时间与版本由
provenance 自动记录，最终结果由 validation 状态决定。

## 5. 自动创建项目

以只优化电荷的分子晶体为例：

```bash
ffopt init btah_charge_only \
  --bulk-data BTAH_bulk.data \
  --single-data BTAH_single.data \
  --cells 8 2 2 \
  --mode charge_only \
  --target a=4.2422,1.0,A \
  --target b=17.8270,1.0,A \
  --target c=20.6850,1.0,A \
  --target alpha=72.63,0.5,degree \
  --target beta=87.15,0.5,degree \
  --target gamma=86.23,0.5,degree \
  --target density=1.3285,1.0,g/cm3 \
  --target sublimation=98.5,0.3,kJ/mol
```

常用 `--mode`：

- `full`：优化 epsilon、sigma、q。
- `fix_sigma`：固定 sigma，优化 epsilon 和 q。
- `charge_only`：固定 epsilon 和 sigma，只优化 q。
- `lj_only`：固定 q，优化 epsilon 和 sigma。

`ffopt init` 会读取 data 中的初始参数并写成显式 `type` 表，同时复制 data 到
规范目录。它无法判断初始值是否真的是 CHARMM，也无法替用户判断参数范围是否
有物理意义；生成后必须人工审核。

初始化器默认写入 `range charge delta 0.30` 和 `charge_limit 2.0`。前者是相对
每个初始电荷的局域 `+/-0.30 e` 搜索窗，后者是所有最终电荷都必须满足的独立
绝对安全上限；对中性有机分子可按物理判断把后者收紧到 `1.0`，二者不能混为一谈。

`--target` 的完整格式为
`NAME=VALUE[,WEIGHT[,UNIT[,TOLERANCE]]]`。例如
`density=1.3285,1.0,g/cm3,0.03`。省略 tolerance 时，`ffopt init` 会把当前默认值
明确写进 `ffopt.in`，用户仍应根据实验误差和拟合用途审核它。

## 6. 编辑 `ffopt.in`

### 6.1 顶层

```text
ffopt 1
project btah_charge_only
workflow bo sample nn al audit finalize validate
```

- `ffopt 1` 是输入格式版本。
- `project` 只是项目和结果目录名，不会调用任何 BTAH 专用代码。
- `workflow bo` 只跑 BO。
- `workflow validate` 直接验证 type 行中的初始参数。
- `workflow bo sample nn al audit finalize validate` 跑完整生产流程。

仅有 `workflow validate` 时不需要写任何 `range`：软件不会建立搜索空间，所有
`type` 初值直接交给 LAMMPS。只要 workflow 包含 `bo`，每个未固定参数就必须有
全局或逐 type 的范围。

### 6.2 参数和范围

```text
parameters
    range epsilon factor 0.50 2.00
    range sigma   factor 0.85 1.15
    range charge  delta  0.30
    charge_limit 1.0
    neutrality derive bhN1
    mixing epsilon geometric
    mixing sigma geometric
    fix epsilon sigma

    # id label epsilon(kcal/mol) sigma(A) charge(e)
    type 1 bhN1 0.2000 3.2963 -0.6220
    type 2 bhHn 0.0474 0.3981  0.4760
end
```

- `factor 0.50 2.00` 表示初始值的 0.5 到 2.0 倍。
- `delta 0.30` 表示初始值加减 0.30。
- `absolute LOW HIGH` 表示直接给绝对上下界。
- `charge_limit 1.0` 表示任何最终电荷都必须满足 `|q| <= 1.0 e`。
- `fix epsilon sigma` 表示只优化电荷。
- 没有 `fix` 表示三类参数全部放开。
- `range type bhN1 charge delta 0.10` 可以覆盖某一个 type 的默认范围。

`neutrality derive bhN1` 会把该 type 的电荷从独立维度中移除，并根据每个 type
的原子数恢复出严格中性的电荷。因此 14 个 type 的 charge-only 通常是 13 维，
但恢复出的第 14 个电荷仍会进入 LAMMPS 和 ANN 特征。

### 6.3 混合规则

当前 LAMMPS 后端固定使用 epsilon 几何平均；sigma 可以选择几何或算术平均：

```text
mixing epsilon geometric
mixing sigma geometric
```

```text
epsilon_ij = sqrt(epsilon_i * epsilon_j)
sigma_ij = sqrt(sigma_i * sigma_j)          # geometric
sigma_ij = (sigma_i + sigma_j) / 2          # arithmetic
```

应选择原始力场采用的规则，不要为了获得更低 objective 随意切换。

### 6.4 体相性质

```text
property bulk
    data data/bulk/BTAH_bulk.data
    cells_in_data 8 2 2
    temperature 300 K
    pressure 1 atm
    equilibration 20000
    production 40000
    target a 4.2422 A weight 1.0 tolerance 0.15
    target density 1.3285 g/cm3 weight 1.0 tolerance 0.03
end
```

bulk 标准流程写死为：固定盒子最小化、生成速度、三斜全柔性 NPT 平衡、NPT
生产统计。`cells_in_data` 是 data 文件中已经包含的晶胞重复数，不会再复制超胞。

可修改且真正生效的 bulk 参数如下：`temperature`（默认 `300 K`）、`pressure`
（默认 `1 atm`）、`timestep`（默认 `1 fs`）、`cutoff`（默认 `8 A`）、
`equilibration`（默认 `20000` 步）、`production`（默认 `40000` 步）和速度种子
`seed`（默认 `101`）。schema 1 不提供 bulk `protocol` 开关，不能把标准流程改成
NVT 或仅最小化；`production` 至少为固定统计间隔所需的 `5000` 步。

### 6.5 升华焓目标

```text
property sublimation
    bulk data/bulk/BTAH_bulk.data
    single data/molecule/BTAH_single.data
    temperature 298.15 K
    target 98.5 kJ/mol weight 0.3 tolerance 5.0
end
```

当前计算量为：

```text
E_sub,estimate = E_single,min - <PE_bulk,NPT> / N_molecules
```

它用势能差近似实验有限温度升华焓，没有加入理想气体平动、转动、振动、pV 和
标准态热修正。因此软件输出必须保留该计算定义。目标仍可使用实验升华焓，但用户
需要理解这里是受控近似，而不是完整热化学自由能计算。

这里的 `temperature`（默认 `298.15 K`）记录实验目标温度；真正的体相模拟温度由
bulk 模块控制，默认 `300 K`。可选 `cutoff` 只覆盖单分子最小化截断，省略时继承
bulk 截断。每个分子的原子数直接从必须提供的 single data 文件读取，不能另外手写。

### 6.6 吸附

```text
property adsorption
    data complex data/adsorption/BTAH_Au111_complex.data
    data slab data/adsorption/BTAH_Au111_slab.data
    data molecule data/adsorption/BTAH_Au111_molecule.data
    protocol minimize
    metal Au
end
```

没有 `target` 时，吸附能只在最终验证计算，不进入 BO、ANN 和 AL。存在可靠实验值
时才加入：

```text
target -3.5 kcal/mol weight 1.0 tolerance 0.5
```

吸附模块当前固定为确定性的 0 K 最小化。除必需路径和 `protocol minimize` 外，只有
`metal`（默认 `Au`）和 `cutoff`（默认 `7 A`）可修改；温度、时间步、随机种子、
平衡步数和生产步数不会被接受，避免用户写了参数却实际不生效。

`metal` 必须对应一个固定、零电荷的基底 type，其 LJ 参数继续使用各 data 文件中
的值；FFOpt 只更新 complex 与孤立分子共同拥有的分子 types。不要把带电、多组分
或也需要拟合参数的基底强行改名为一个 metal type 来绕过这一约束。

### 6.7 目标函数

每个性质的相对误差为：

```text
r_i = |calculated_i - target_i| / |target_i|
```

总目标函数：

```text
J = sqrt(sum(w_i * r_i^2) / sum(w_i))
```

`weight` 决定该性质对优化的相对贡献。`tolerance` 是最终验证时的绝对误差门槛，
不参与 objective 计算。单位只做严格验证，不自动换算；当前应使用文档列出的
`A`、`degree`、`g/cm3`、`kJ/mol` 和 `kcal/mol`。

### 6.8 BO

```text
bo
    method auto
    initial_points 48
    batch_size 48
    max_rounds 200
    random_seed 42
    stability_audit on
    stability_top_k 20
    stability_seeds 101 202 303
    early_stop patience 30
end
```

- `method auto` 根据独立维度选 GP、TuRBO 或 SAASBO。`ffopt explain` 和
  `ffopt doctor` 会在运行前显示最终方法；若高维任务本应使用 SAASBO 但没有安装
  Pyro，会明确警告并显示回退到 TuRBO，而不是只写一个含糊的 `auto`。
- `initial_points` 是初始 Latin hypercube（LHS）设计点数。`type` 行给出的完整
  初始参数还会作为一个 warm-start 中心额外计算一次，因此
  `initial_points 48` 对应初始阶段共 49 次 LAMMPS 评估。
- `batch_size` 是每轮科学候选数，不能与机器 `workers` 混为一谈。
- `max_rounds` 是最大轮数。
- `patience` 表示连续多少轮没有达到最小改进后提前结束。
- BO 稳定性审计会对最优候选换种子验证，降低偶然低 objective 的风险。
- `stability_top_k 20` 和三个 `stability_seeds` 表示搜索结束后最多额外执行
  `20 * 3 = 60` 次 LAMMPS；它们控制审计成本，不是 NN 截断百分比。
- `ffopt explain` 会分别打印 LHS 点数、warm-start 中心数、初始总数和完整 BO
  搜索预算，还会单列稳定性审计上限与 BO 阶段总上限，避免把机器并发数误认为
  科学采样数。

高级阈值 `stability_max_objective_std`、`stability_max_property_rel_std`、
`stability_noise_penalty` 和 `stability_failure_penalty` 默认分别为 `0.05`、`0.05`、
`1.0`、`10.0`。没有针对新材料的噪声依据时不建议随意修改。

### 6.9 局域采样

```text
sample
    points 2000
    centers 24
    center_selection diverse
    radii 0.01 0.025 0.05
    global_fraction 0.10
    seeds 101 202 303
end
```

- `points` 是独立参数向量数。
- 三个 `seeds` 表示每个向量计算三次，总评估数为 `points * 3`。
- `centers` 是从 BO 稳定结果中选择的多个中心。
- `radii` 是相对于完整参数上下界的归一化半径，不是 `e`。
- `global_fraction` 是从完整范围补充的比例。

多中心局域数据用于学习多个稳定盆地；全局数据用于识别边界和失败区域，但过多
宽域、不稳定数据会降低低 objective 局域精度。

### 6.10 ANN

```text
nn
    method ann
    ensemble 8
    hidden_layers 256 128 128 64
    epochs 1200
    batch_size 128
    learning_rate 0.0005
    validation_fraction 0.15
    test_fraction 0.10
end
```

`ensemble 8` 表示训练 8 个独立初始化的 ANN。预测值取均值，模型间分歧用于估计
认知不确定性。模型学习的是本材料中“独立参数向量到 LAMMPS 性质”的映射，不是
跨材料通用的图神经网络。必须逐性质检查测试集 R2、RMSE、残差和低 objective
区域覆盖，不能只看一个总 R2。

### 6.11 主动学习

```text
al
    acquisition uncertainty
    rounds 2
    candidates 20
    candidate_pool 16384
    sampling_domain core_envelope
end
```

每轮先由代理模型筛选候选，再用真实 LAMMPS 计算 selected candidates，加入训练集
后重新学习。最终好坏以 LAMMPS 验证 objective 为准，不以 ANN 预测值为准。

### 6.12 稳健审计与最终定稿

```text
audit
    top_k 8
    seeds 101 202 303
end
```

`audit` 从截至 AL 的累计数据中取互不重复、objective 最低的 `top_k` 个候选，
每个候选再按列出的种子运行 LAMMPS。上例共进行 `8 * 3 = 24` 次评估，并按
`objective 均值 + objective 标准差` 排序。失败种子不会被悄悄删除，而会保留在
复算表中。

BO 内部稳定性审计和这里的最终审计用途不同：前者为后续局域采样选择可重复的
BO 中心；后者审核 AL 后的候选并决定最终可交付参数。`finalize` 没有参数块，只需
写在 workflow 中；它会解析全部固定参数和中性约束恢复电荷，并导出完整 type 表。

### 6.13 最终验证

```text
validate
    require_tolerances yes
    objective_max 0.03
    max_error_percent 3.0
end
```

三个门槛相互独立：总 objective、最大相对误差、每个性质的绝对 tolerance。最终
验证默认且始终保存性质表、参数表、日志、最终结构和相应轨迹，不需要额外写轨迹
开关。没有科学依据时不要照抄 BTAH 的阈值，应按新材料实验误差和模型目标设置。

## 7. 运行前四步检查

```bash
cd my_project
ffopt check ffopt.in
ffopt explain ffopt.in
ffopt doctor ffopt.in --machine cluster-1node
ffopt run ffopt.in --machine cluster-1node --dry-run
```

确认：

1. 独立维度与预期一致。
2. 固定、自由和中性恢复参数正确。
3. 拟合性质与“仅最终验证性质”没有混淆。
4. data 路径、LAMMPS、MPI、PyTorch 和 SLURM 全部为 OK。
5. dry-run 中阶段顺序与 `workflow` 一致。

## 8. 正式运行和自动续算

本地：

```bash
ffopt run ffopt.in --machine local-workstation
```

SLURM 自动跟进：

```bash
ffopt run ffopt.in --machine cluster-1node --watch
```

任务被 wall time 杀死、SSH 断开或节点失败后，重复完全相同的命令：

```bash
ffopt run ffopt.in --machine cluster-1node --watch
```

默认 `.in` 项目会自动读取 `state.sqlite`，检查作业状态、checkpoint 和必须产物，
从第一个不完整阶段恢复。不要使用 `--new` 续算；`--new` 明确表示新建独立计算。
同一个 run 已经产生阶段记录后，必须使用同一 FFOpt 版本续算；软件检测到版本变化
会拒绝混用结果。此时应恢复原版本，或确认要重新开始后使用 `--new`。

暂时只跑到 BO：

```bash
ffopt run ffopt.in --machine cluster-1node --until bo --watch
```

以后继续：

```bash
ffopt run ffopt.in --machine cluster-1node --from-stage sample --watch
```

长期只需要 BO 时，直接把输入写成 `workflow bo` 更清楚。

## 9. 查看状态、日志和结果

```bash
ffopt status ffopt.in --machine cluster-1node
ffopt logs ffopt.in --stage bo --lines 120
ffopt results ffopt.in
squeue -u "$USER"
```

在 SLURM 主机上，`ffopt status` 会显示活动作业的实时 `RUNNING/PENDING` 状态、
节点或排队原因；BO 已写 checkpoint 时还会显示已完成轮次、评估数和当前最佳
objective。因此即使阶段最终 CSV 尚未生成，也能判断是否具备续算点。对于已经
存在的 pipeline，省略 `--machine` 时会显示状态库中实际记录的 profile，不会把
集群结果误标为 `local`。

核心结果目录：

```text
runs/<project>/pipelines/default/
```

重点文件：

- `bo/all_results.csv`：所有 BO 参数、性质、objective 和成功状态。
- `bo/stable_results.csv`：换种子后的稳健排序。
- `sample/local_replicates.csv`：每个参数点、每个种子的原始结果。
- `sample/local_results.csv`：聚合后的机器学习训练表。
- `nn/forward_nn.pt`：ANN 集成模型和特征元数据。
- `nn/parity_data.csv`：测试集真实值与预测值。
- `nn/nn_optimize_result.json`：逐性质指标和 NN 候选。
- `al/active_learning_history.json`：每轮 AL 预测与 LAMMPS 结果。
- `al/final_parameters.json`：AL 阶段最终参数。
- `audit/stability_replicates.csv`：最终候选逐种子的原始复算结果。
- `audit/stable_results.csv`：最终候选的均值、标准差和稳健 objective 排名。
- `finalize/final_summary.json`：稳健选择规则、来源和最终自由/派生参数。
- `finalize/final_parameters.lammps`：稳健定稿后的完整 LAMMPS 参数命令。
- `validate/computed_properties.csv`：最终性质、目标、误差和 tolerance。
- `validate/validation_summary.json`：最终验收结论和失败原因。
- `validate/final_atom_parameters.csv`：全部 type 的最终 epsilon、sigma 和 q。
- `validate/final_parameters.lammps`：可在 `read_data` 后 include 的完整 LAMMPS 命令。
- `validate/final_parameters.json`：自由、固定、派生参数及来源。
- `validate/eval_0000/bulk/`、`validate/eval_0000/adsorption/`：最终结构、日志和轨迹。

## 10. 单节点与双节点如何比较

必须使用完全相同的 `ffopt.in`、同一个软件版本和相同随机种子。分别使用两个机器
profile 运行软件验收：

```bash
ffopt self-test --machine cluster-1node \
  --workdir "$HOME/ffopt-acceptance/one-node" --watch

ffopt self-test --machine cluster-2node \
  --workdir "$HOME/ffopt-acceptance/two-node" --watch
```

验收时比较：

- 两边最终 validation 必须 PASS。
- `batch_size`、初始点、采样 points 和 seeds 必须相同。
- 最终性质和 objective 应在随机模拟合理波动内一致。
- 双节点主要缩短 BO、sample 和 AL 中并行 LAMMPS 的墙钟时间。
- 如果双节点给出不同科学预算，说明配置或软件存在 bug，而不是“更多节点更准确”。

## 11. 常见问题

### `torch.cuda.is_available()` 为 False

CPU 版 PyTorch 的正常输出。CPU profile 可直接运行。需要 GPU 时另建 CUDA 环境，
不要污染 base。

### `^M` 是否影响 LAMMPS

通常只是 CRLF 的显示。先运行 `ffopt inspect` 和 `lmp -in ...` 验证；解析无误时
无需为了美观修改生产源文件。

### `DependencyNeverSatisfied`

表示某个 SLURM 依赖任务失败、取消或未满足，后继任务不会启动。新 pipeline
runner 不依赖用户手写一串 `afterok`；重复 `ffopt run ... --watch` 会检查阶段产物
并重新提交不完整阶段。

### 作业一直是 `PD`

`Resources` 表示当前没有足够的节点/CPU/内存同时满足请求，作业仍可在资源释放后
启动；`Priority` 表示正在等待调度优先级或回填窗口；`QOSMaxCpuPerJobLimit` 表示
`--total-cores` 超过该 QOS 允许的单作业 CPU 上限，不会靠等待自行解决。前两者通常
先等待或缩短验收 walltime，最后一种必须降低同名 machine profile 的 nodes、workers
和 total-cores，或向管理员申请合适 QOS，再用 `machine configure --force` 更新。
修改机器并发不会改变 `ffopt.in` 中的 BO/sample 科学预算。

### BO objective 很好但 NN R2 低

BO 点是自适应相关数据，不一定覆盖平滑映射；稳定点数量可能不足，某些性质动态
范围太小，或者不同种子噪声大于参数响应。应先增加多中心局域采样和稳定标签，
而不是单纯加深网络。

### 全局 R2 差、局域 R2 好

说明当前 LAMMPS 响应只在稳定局域可学习，或全局包含结构转变和失败边界。ANN
不能凭空学习未覆盖的稳定全局关系，也不能可靠外推。扩大范围时应分层采样，保留
局域高精度模型，并显式加入边界/失败数据分析。

### 固定参数后 objective 无法继续降低

机器学习可以准确学习一个“无法到达实验目标”的映射。表达能力不足影响可达到的
最优 objective，不等于必然导致低 R2；低 R2 还要分别检查数据量、噪声、性质方差、
采样覆盖和模型。两类问题不能混为一谈。

## 12. 发布和论文归档检查表

保留：

1. 原始 data 文件和生成方式。
2. 最终 `ffopt.in`。
3. 完整 selected pipeline 目录。
4. FFOpt、LAMMPS、Python、PyTorch 和 MPI 版本。
5. 使用的 `machines.toml` profile（删除私密环境信息后）。
6. BO、采样、ANN、AL 和最终验证表。
7. 最终参数、结构和轨迹。
8. 实验目标来源、单位、温度和不确定度。
9. 升华焓近似公式及未包含的热修正。
10. 所有人工排除、固定参数和范围选择的依据。

不要手工修改生成 CSV 后继续计算，也不要只保留截图而丢失机器可读结果。

## 13. 下一步阅读

- [`ffopt.in` 逐项参考](../reference/input-file.md)
- [`0.3.0a3` 单/双节点真实验收记录](../reference/acceptance-v0.3.0a3.md)
- [机器配置参考](../reference/machine-profiles.md)
- [输出与命名参考](../reference/outputs-and-naming.md)
- [工作流与精度原理](../explanation/workflow-and-accuracy.md)
- [BTAH 示例](../../examples/btah/README.md)
