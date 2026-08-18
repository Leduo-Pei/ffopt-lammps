# FFOpt-LAMMPS 完整中文使用手册

本手册面向第一次开发分子力场参数的用户。目标是：用户只准备
LAMMPS data 文件、实验目标和初始参数范围，编辑一个 `ffopt.in`，即可完成
BO、局域采样、ANN、主动学习和最终 LAMMPS 验证，并在中断后自动续算。

> 当前版本为 alpha。正式支持范围是分子晶体、孤立分子和分子吸附模型。
> BTAH 是软件回归体系。单质、合金、反应力场和多晶型迁移尚未作为当前版本
> 的通用能力承诺。

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

```bash
conda create -n ffopt python=3.11 -y
conda activate ffopt
conda install -c conda-forge lammps openmpi -y

python -m pip install \
  "ffopt-lammps[full] @ git+https://github.com/Leduo-Pei/ffopt-lammps.git@v0.3.0a2"
```

检查安装来源：

```bash
which python
which ffopt
which lmp
which mpirun
python -c "import ffopt, torch; print(ffopt.__version__); print(torch.__version__, torch.cuda.is_available())"
lmp -help | head
```

CPU 环境中 `torch.cuda.is_available()` 显示 `False` 是正常现象，不是报错。
这表示 NN/AL 使用 CPU。只有安装 CUDA 版 PyTorch 且能识别 GPU 时才会显示
`True`。

### 2.3 Windows 本地安装

在 Anaconda Prompt 或 PowerShell 中：

```powershell
conda create -n ffopt python=3.11 -y
conda activate ffopt
python -m pip install "ffopt-lammps[full] @ git+https://github.com/Leduo-Pei/ffopt-lammps.git@v0.3.0a2"
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

## 3. 配置机器

### 3.1 自动探测

```bash
ffopt machine probe
ffopt machine probe --partition CPU
```

该命令只读取环境，不修改配置。它会报告 Python、LAMMPS、MPI、CPU、GPU 和
SLURM 分区，并给出保守建议。建议仍需结合本集群的节点共享和内存规则审核。

### 3.2 本地机器

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

多次执行 `machine configure` 时，不加 `--force` 会拒绝覆盖；加 `--force`
只替换同名 profile，不会追加重复表，也不会删除其他机器配置。

### 3.5 机器验收

```bash
ffopt machine list
ffopt machine show --name cluster-1node
ffopt machine test --name cluster-1node
ffopt self-test --machine cluster-1node --watch
```

`machine test` 只检查一个极小 LAMMPS/MPI 作业。`self-test` 会跑完整 BTAH
软件验收流程，并检查最终 objective、性质误差和 tolerance；新服务器正式计算前
必须先通过两者。

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

## 6. 编辑 `ffopt.in`

### 6.1 顶层

```text
ffopt 1
project btah_charge_only
workflow bo sample nn al validate
```

- `ffopt 1` 是输入格式版本。
- `project` 只是项目和结果目录名，不会调用任何 BTAH 专用代码。
- `workflow bo` 只跑 BO。
- `workflow validate` 直接验证 type 行中的初始参数。
- `workflow bo sample nn al validate` 跑完整流程。

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
    early_stop patience 30
end
```

- `method auto` 根据独立维度选 GP、TuRBO 或 SAASBO。
- `initial_points` 是初始设计数量。
- `batch_size` 是每轮科学候选数，不能与机器 `workers` 混为一谈。
- `max_rounds` 是最大轮数。
- `patience` 表示连续多少轮没有达到最小改进后提前结束。
- BO 稳定性审计会对最优候选换种子验证，降低偶然低 objective 的风险。

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

### 6.12 最终验证

```text
validate
    trajectory final
    require_tolerances yes
    objective_max 0.03
    max_error_percent 3.0
end
```

三个门槛相互独立：总 objective、最大相对误差、每个性质的绝对 tolerance。最终
验证保存性质表、参数表、日志、最终结构和相应轨迹。没有科学依据时不要照抄 BTAH
的阈值，应按新材料实验误差和模型目标设置。

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
- `validate/computed_properties.csv`：最终性质、目标、误差和 tolerance。
- `validate/validation_summary.json`：最终验收结论和失败原因。
- `validate/bulk/`、`validate/adsorption/`：结构、日志和轨迹。

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
- [机器配置参考](../reference/machine-profiles.md)
- [输出与命名参考](../reference/outputs-and-naming.md)
- [工作流与精度原理](../explanation/workflow-and-accuracy.md)
- [BTAH 示例](../../examples/btah/README.md)
