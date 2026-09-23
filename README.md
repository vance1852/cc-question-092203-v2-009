# 风电场布局优化工具

这个项目用于估算风电场的年发电量，并比较不同风机布局和尾流模型的结果。项目包含风机与风资源模型、场地边界和间距约束、遗传算法与粒子群优化、经济性分析以及无界面图表输出。

## 安装

建议使用 Python 3.10 或更新版本，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 可以使用 `.venv\\Scripts\\Activate.ps1` 激活环境。

## 快速验证

```bash
python quick_test.py
```

快速验证会覆盖模型、约束、年发电量、优化、经济性和图表生成，并在 `test_output/` 写入临时图片。该目录不会纳入版本控制。

## 完整分析

```bash
python -m wind_farm_opt --help
python -m wind_farm_opt --n-turbines 15 --iterations 100 --population 50 --output-dir output
```

也可以先生成配置文件，再通过 `--config` 运行：

```bash
python -m wind_farm_opt --generate-config my_config.json
python -m wind_farm_opt --config my_config.json
```

所有运行结果默认写入 `output/`，可以用 `--no-plots` 跳过图表生成。命令行使用无界面绘图后端，适合容器和服务器环境。

## 方向性（椭圆）间距约束

默认沿用各向同性的径向最小间距（`min_spacing_multiple`，默认 5D）。
场址审批若沿主风向、横风向规定了不同的安全间距，可启用方向性椭圆安全域：
每对机组在以参考风向为轴的旋转坐标系中判定
`(顺风分量 / (顺风倍数·D))² + (横风分量 / (横风倍数·D))² ≥ 1`，
其中 `D` 取机对两台机组转子直径的较大值。椭圆恰好过边界的点视为可接受
（统一相对容差 `1e-9`），零距离必然违规且报告有限的所需净距。

命令行：

```bash
python -m wind_farm_opt \
  --directional-spacing \
  --spacing-direction 270 \
  --downwind-multiple 7 \
  --crosswind-multiple 3
```

- `--spacing-direction`：参考风向（度，气象惯例 0=北、顺时针），如 `270`
  表示西风，顺风轴沿流向指向东；
- `--downwind-multiple` / `--crosswind-multiple`：顺风 / 横风安全间距倍数；
- 配置文件对应字段位于 `optimization` 节：
  `directional_spacing`、`spacing_reference_direction`、
  `downwind_spacing_multiple`、`crosswind_spacing_multiple`；
- 用 `--no-directional-spacing` 可在命令行上覆盖配置文件、强制回到径向模式；
- 规则网格、交错布局、间距修复、GA 与 PSO 均在同一约束下生成结果，
  拥挤场地无法满足时抛出 `SpacingFeasibilityError` 有界退出；
- 布局图默认在首台机组上绘制代表性安全域，可用 `--no-safety-zones` 关闭
  （或配置 `visualization.plot_safety_zones`）。

`check` 返回的每个违规机对包含顺风/横风方向分量、要求的顺风/横风净距与
沿当前连线仍需拉开的净距，可直接用于审批反馈或修复。
