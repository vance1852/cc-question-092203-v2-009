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

方向性椭圆间距的专项验证：

```bash
python test_directional_spacing.py
```

## 方向性间距约束

默认使用径向（圆形）最小间距 `--min-spacing`（转子直径倍数）。场址审批沿主风向给出不同安全间距时，可启用方向性椭圆安全域：每对机组在随参考风向旋转的坐标系中，分别按顺风、横风倍数判断椭圆安全域

$$\left(\frac{s_{\parallel}}{k_{\downarrow}D}\right)^2+
\left(\frac{s_{\perp}}{k_{\times}D}\right)^2 \ge 1$$

恰好在安全域边界上的机对判为合格，重合机组（零距离）始终判为违规。

```bash
python -m wind_farm_opt \
  --directional-spacing \
  --reference-direction 270 \
  --downwind-multiple 7 \
  --crosswind-multiple 3 \
  --show-safety-zones \
  --n-turbines 15 --output-dir output
```

- `--reference-direction`：参考风向（度，气象习惯：风的来向），椭圆长轴沿气流方向并随之旋转；
- `--downwind-multiple` / `--crosswind-multiple`：顺风 / 横风安全间距倍数（转子直径倍数）；
- `--show-safety-zones`：在布局图中绘制若干代表性机组的椭圆（方向性）或圆形（径向）安全域；
- `--no-directional-spacing`：关闭新规则，强制回到径向圆形间距。

规则网格、交错布局、间距修复、GA 与 PSO 都只生成满足该约束的布局；拥挤场地无法满足时会在有限次尝试后报错退出，不会返回违规结果。固定 `--seed` 后，方向性与径向两种模式均可重复。配置文件中对应字段为 `optimization.directional_spacing.{enabled,reference_direction,downwind_multiple,crosswind_multiple}`。

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
