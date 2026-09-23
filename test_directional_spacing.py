"""方向性椭圆间距约束的功能验证脚本。"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from wind_farm_opt.constraints.boundary import create_rectangular_boundary
from wind_farm_opt.constraints.spacing import (
    SpacingConstraint,
    PairViolation,
    SPACING_TOLERANCE,
    ZERO_DISTANCE,
    enforce_spacing,
    check_min_spacing,
)
from wind_farm_opt.optimization.baseline import (
    generate_grid_layout,
    generate_staggered_grid_layout,
)
from wind_farm_opt.optimization.ga import GeneticAlgorithm, GAConfig
from wind_farm_opt.optimization.pso import ParticleSwarmOptimizer, PSOConfig
from wind_farm_opt.visualization.plotting import plot_farm_layout

D = 100.0
diams = np.full(6, D)
ok_all = True


def check(name, cond, detail=""):
    global ok_all
    status = "✓" if cond else "✗"
    if not cond:
        ok_all = False
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


print("1. 椭圆几何：圆形规则拒绝的横风近距排布在方向性规则下可接受")
c = SpacingConstraint.directional(diams, reference_direction=270.0,
                                  downwind_multiple=7.0, crosswind_multiple=3.0)
# 270° 来风（西风），气流沿 +x。横风 = y 方向。
pos_cross = np.array([[0.0, 0.0], [0.0, 3.05 * D]])  # 横风 3.05D < 圆形 5D
valid_rad, _ = check_min_spacing(pos_cross, 5.0 * D)
valid_dir, viol = c.check(pos_cross)
check("圆形 5D 规则拒绝该排布", not valid_rad)
check("方向性规则接受横风 3.05D 排布", valid_dir and len(viol) == 0)

pos_down = np.array([[0.0, 0.0], [6.5 * D, 0.0]])  # 顺风 6.5D < 7D
valid, viol = c.check(pos_down)
check("方向性规则拒绝顺风 6.5D 排布", not valid and len(viol) == 1)
v = viol[0]
check("违规报告含顺风分量", abs(v.along - 6.5 * D) < 1e-6,
      f"along={v.along:.1f} cross={v.cross:.1f}")
check("违规报告横风分量≈0", abs(v.cross) < 1e-6)
check("报告所需顺风净距=7D", abs(v.required_along - 7 * D) < 1e-6)
check("报告所需横风净距=3D", abs(v.required_cross - 3 * D) < 1e-6)
check("沿当前方位所需净距≈7D", abs(v.required_distance - 7 * D) < 1e-6)

pos_cross_bad = np.array([[0.0, 0.0], [0.0, 2.5 * D]])
valid, viol = c.check(pos_cross_bad)
check("横风 2.5D 违规且所需净距=3D",
      not valid and abs(viol[0].required_distance - 3 * D) < 1e-6)

print("\n2. 随参考方向旋转：物理方位判定与坐标系旋转一致")
c90 = SpacingConstraint.directional(diams, reference_direction=180.0,
                                    downwind_multiple=7.0, crosswind_multiple=3.0)
# 180° 南风（气象约定：风从南来），气流沿 +y；原沿 x 的 6.5D 对现在
# 是纯横风 → 应接受
p = np.array([[0.0, 0.0], [6.5 * D, 0.0]])
check("参考方向旋转后 x 向 6.5D 变为横风（可接受）", c90.check(p)[0])
p2 = np.array([[0.0, 0.0], [0.0, 6.5 * D]])
check("旋转后 +y 向 6.5D 为顺风（拒绝）", not c90.check(p2)[0])
# 同方位对角对：两个互成 90° 的参考方向，顺/横分量互换（轴反向带符号）
pdiag = np.array([[0.0, 0.0], [2.5 * D, 1.5 * D]])
v270 = c.check(pdiag)[1][0]
v180 = c90.check(pdiag)[1][0]
check("两种参考方向下该对角对均违规", v270 is not None and v180 is not None)
check("旋转 90° 后顺/横分量互换",
      abs(v270.along + v180.cross) < 1e-6 and abs(v270.cross - v180.along) < 1e-6,
      f"({v270.along:.0f},{v270.cross:.0f}) vs ({v180.along:.0f},{v180.cross:.0f})")

print("\n3. 边界容差与零距离的一致定义")
# 恰好在椭圆边界上（q=1）判为合格
pb = np.array([[0.0, 0.0], [0.0, 3.0 * D]])
check("恰在椭圆边界（横风 3D）判合格", c.check(pb)[0])
pa = np.array([[0.0, 0.0], [7.0 * D, 0.0]])
check("恰在椭圆边界（顺风 7D）判合格", c.check(pa)[0])
# 圆形边界点判合格
cr = SpacingConstraint.radial(diams, min_multiple=5.0)
check("圆形恰在 5D 边界判合格", cr.check(np.array([[0, 0], [5 * D, 0]]))[0])
# 零距离：任何非零安全域下违规，且不抛异常、所需净距有定义
pz = np.array([[0.0, 0.0], [0.0, 0.0]])
for mode_name, cc in [("方向性", c), ("径向", cr)]:
    valid, vz = cc.check(pz)
    check(f"{mode_name}零距离判违规", not valid and len(vz) == 1)
    check(f"{mode_name}零距离所需净距>0且距离=0",
          vz[0].distance == 0.0 and vz[0].required_distance > 0.0)

print("\n4. 混合转子直径：逐对依据自身（较大）直径")
dmix = np.array([100.0, 200.0])
cmix = SpacingConstraint.directional(dmix, 270.0, 7.0, 3.0)
# 与 D=200 机组相邻：横风需 3*200=600
pmix = np.array([[0.0, 0.0], [0.0, 550.0]])
valid, vm = cmix.check(pmix)
check("大小机对按大直径(200)判定违规", not valid
      and abs(vm[0].binding_diameter - 200.0) < 1e-9
      and abs(vm[0].required_cross - 600.0) < 1e-9)
check("横风 600 恰好合格", cmix.check(np.array([[0, 0], [0, 600.0]]))[0])

print("\n5. 规则网格 / 交错布局：方向性模式下全部合规")
boundary = create_rectangular_boundary(4000, 4000)
diams12 = np.full(12, D)
c12 = SpacingConstraint.directional(diams12, 270.0, 7.0, 3.0)
cr12 = SpacingConstraint.radial(diams12, min_multiple=5.0)
rng = np.random.default_rng(7)
g = generate_grid_layout(boundary, 12, diams12, rng=rng, spacing_constraint=c12)
check("规则网格满足方向性约束", c12.is_satisfied(g))
check("规则网格全部在场内", bool(boundary.contains_all(g).all()))
s = generate_staggered_grid_layout(boundary, 12, diams12, rng=rng,
                                   spacing_constraint=c12)
check("交错网格满足方向性约束", c12.is_satisfied(s)
      and bool(boundary.contains_all(s).all()))
gr = generate_grid_layout(boundary, 12, diams12, min_multiple=5.0,
                          rng=np.random.default_rng(7))
check("关闭新规则时网格满足径向 5D", cr12.is_satisfied(gr))

print("\n6. 间距修复：违规布局被推开到椭圆外（含重合对）")
c6 = c
bad = np.array([[0.0, 0.0], [2.0 * D, 0.0], [0.0, 2.0 * D],
                [500.0, 500.0], [-500.0, 500.0], [500.0, -500.0]], dtype=float)
fixed = enforce_spacing(bad, c6, boundary, rng=np.random.default_rng(1),
                        max_iterations=2000)
check("修复后满足方向性约束", c6.is_satisfied(fixed))
check("修复后全部在场内", bool(boundary.contains_all(fixed).all()))
coinc = np.zeros((4, 2))
c4 = SpacingConstraint.directional(np.full(4, D), 270.0, 7.0, 3.0)
fixed0 = enforce_spacing(coinc, c4, boundary, rng=np.random.default_rng(2))
check("全部重合的布局也能修复", c4.is_satisfied(fixed0)
      and bool(boundary.contains_all(fixed0).all()))

print("\n7. 拥挤场地：有界退出（不返回违规布局、不无限循环）")
# 200 m 方框对角线仅 283 m，小于横风半轴 300 m，两台机组在几何上
# 也不可能同时合规——确保任何方向都无法满足。
tiny = create_rectangular_boundary(200, 200)
diams10 = np.full(10, D)
c10 = SpacingConstraint.directional(diams10, 270.0, 7.0, 3.0)
try:
    generate_grid_layout(tiny, 10, diams10, rng=np.random.default_rng(3),
                         spacing_constraint=c10)
    check("拥挤网格抛出 RuntimeError", False)
except RuntimeError as e:
    check("拥挤网格抛出 RuntimeError", True, str(e)[:40])
try:
    enforce_spacing(np.array([[0.0, 0.0], [10.0, 10.0]]), c6, tiny,
                    rng=np.random.default_rng(3), max_iterations=50)
    check("拥挤修复抛出 RuntimeError", False)
except RuntimeError:
    check("拥挤修复抛出 RuntimeError", True)
try:
    GeneticAlgorithm(3, np.full(3, D), tiny, lambda p: 1.0,
                     GAConfig(population_size=4, max_generations=2, seed=1,
                              spacing_constraint=SpacingConstraint.directional(
                                  np.full(3, D), 270.0, 7.0, 3.0))
                     ).optimize(verbose=False)
    check("拥挤场地 GA 有界退出", False)
except RuntimeError:
    check("拥挤场地 GA 有界退出", True)

print("\n8. GA / PSO：方向性与径向模式结果均合规")
wr_diams = np.full(10, D)
wc = SpacingConstraint.directional(wr_diams, 270.0, 7.0, 3.0)
fit = lambda p: -np.sum(np.var(p, axis=0))  # 确定性虚拟目标
ga = GeneticAlgorithm(10, wr_diams, boundary, fit,
                      GAConfig(population_size=8, max_generations=4, seed=42,
                               spacing_constraint=wc))
res_ga = ga.optimize(verbose=False)
check("GA 方向性结果合规", wc.is_satisfied(res_ga.best_positions)
      and bool(boundary.contains_all(res_ga.best_positions).all()))
pso = ParticleSwarmOptimizer(10, wr_diams, boundary, fit,
                             PSOConfig(swarm_size=8, max_iterations=4, seed=42,
                                       spacing_constraint=wc))
res_pso = pso.optimize(verbose=False)
check("PSO 方向性结果合规", wc.is_satisfied(res_pso.best_positions)
      and bool(boundary.contains_all(res_pso.best_positions).all()))

ga_r = GeneticAlgorithm(10, wr_diams, boundary, fit,
                        GAConfig(population_size=8, max_generations=4, seed=42))
check("GA 径向模式结果满足 5D",
      SpacingConstraint.radial(wr_diams, 5.0).is_satisfied(
          ga_r.optimize(verbose=False).best_positions))

print("\n9. 固定种子可重复（两种模式）")
def run_ga(seed, directional):
    cc = SpacingConstraint.directional(wr_diams, 270.0, 7.0, 3.0) if directional \
        else SpacingConstraint.radial(wr_diams, 5.0)
    opt = GeneticAlgorithm(10, wr_diams, boundary, fit,
                           GAConfig(population_size=8, max_generations=3,
                                    seed=seed, spacing_constraint=cc))
    return opt.optimize(verbose=False).best_positions
d1, d2 = run_ga(123, True), run_ga(123, True)
r1, r2 = run_ga(123, False), run_ga(123, False)
check("方向性模式同种子结果一致", np.allclose(d1, d2))
check("径向模式同种子结果一致", np.allclose(r1, r2))
d3 = run_ga(124, True)
check("不同种子结果不同", not np.allclose(d1, d3))
g1 = generate_grid_layout(boundary, 12, diams12, rng=np.random.default_rng(9),
                          spacing_constraint=c12)
g2 = generate_grid_layout(boundary, 12, diams12, rng=np.random.default_rng(9),
                          spacing_constraint=c12)
check("网格生成同种子可重复", np.allclose(g1, g2))

print("\n10. 代表性安全域绘图")
os.makedirs("test_output", exist_ok=True)
plot_farm_layout(g, boundary, diams12, spacing_constraint=c12,
                 show_safety_zones=True,
                 save_path="test_output/directional_zones.png", show=False)
plot_farm_layout(gr, boundary, diams12, spacing_constraint=cr12,
                 show_safety_zones=True,
                 save_path="test_output/radial_zones.png", show=False)
check("两张安全域图已生成",
      os.path.exists("test_output/directional_zones.png")
      and os.path.exists("test_output/radial_zones.png"))

print("\n" + "=" * 60)
print("全部方向性间距验证通过! ✓" if ok_all else "存在失败项! ✗")
print("=" * 60)
sys.exit(0 if ok_all else 1)
