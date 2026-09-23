"""基线布局生成（规则网格与交错网格）。

网格与交错布局在约束的旋转坐标系中铺设：方向性椭圆模式下，列沿
参考风向、行沿横风向，行列间距分别不小于顺风/横风半轴，从而保证
相邻与对角机对均在椭圆安全域外；径向模式下退化为世界坐标中的
等距方格。落不到多边形场地内的栅格点由拒绝采样补齐，最后用有界
间距修复收尾；场地过于拥挤、无法在限定尝试次数内得到合规布局时
抛出 :class:`RuntimeError`。
"""

from typing import Optional

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import (
    SpacingConstraint,
    enforce_spacing,
)


def _build_constraint(
    rotor_diameters: np.ndarray,
    min_multiple: float,
    spacing_constraint: Optional[SpacingConstraint],
) -> SpacingConstraint:
    if spacing_constraint is not None:
        return spacing_constraint
    return SpacingConstraint.radial(
        rotor_diameters=rotor_diameters, min_multiple=min_multiple
    )


def _projected_extents(
    boundary: SiteBoundary, e_u: np.ndarray, e_v: np.ndarray
) -> tuple[float, float, float, float]:
    """边界顶点在 (e_u, e_v) 旋转坐标系中的包围范围。"""
    u = boundary.vertices @ e_u
    v = boundary.vertices @ e_v
    return float(u.min()), float(u.max()), float(v.min()), float(v.max())


def _generate_grid(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    constraint: SpacingConstraint,
    staggered: bool,
    aspect_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)

    # 栅格按全场最大直径的安全域铺设（d_max 控制所有机对），
    # 因而即便存在直径差异也天然合规。
    a, b = constraint.semiaxes(float(np.max(rotor_diameters)))
    e_u, e_v = constraint.axes()

    n_rows = max(1, int(np.round(np.sqrt(n_turbines / aspect_ratio))))
    n_cols = max(1, int(np.ceil(n_turbines / n_rows)))

    u_min, u_max, v_min, v_max = _projected_extents(boundary, e_u, e_v)
    margin = 0.5 * min(a, b)
    u_range = max((u_max - u_min) - 2.0 * margin, 0.0)
    v_range = max((v_max - v_min) - 2.0 * margin, 0.0)

    # 目标间距不超过 1.5 倍半轴；场地放不下时取半轴本身（栅格会
    # 超出多边形，超界点改由拒绝采样补齐），绝不压缩到安全域内。
    fit_u = u_range / max(n_cols - 1, 1)
    fit_v = v_range / max(n_rows - 1, 1)
    spacing_u = max(a, min(fit_u, a * 1.5)) if n_cols > 1 else a
    spacing_v = max(b, min(fit_v, b * 1.5)) if n_rows > 1 else b

    origin_u = u_min + margin + (u_range - spacing_u * (n_cols - 1)) / 2.0
    origin_v = v_min + margin + (v_range - spacing_v * (n_rows - 1)) / 2.0

    positions: list[np.ndarray] = []

    def try_add(world_pos: np.ndarray) -> bool:
        world_pos = np.asarray(world_pos, dtype=np.float64)
        if not boundary.contains_point(world_pos):
            return False
        if positions:
            placed = np.array(positions, dtype=np.float64)
            if not constraint.candidate_acceptable(placed, world_pos):
                return False
        positions.append(world_pos)
        return True

    for row in range(n_rows):
        offset = 0.5 * spacing_u if (staggered and row % 2 == 1) else 0.0
        for col in range(n_cols):
            if len(positions) >= n_turbines:
                break
            u = origin_u + col * spacing_u + offset
            v = origin_v + row * spacing_v
            try_add(u * e_u + v * e_v)

    # 拒绝采样补齐：批量尝试，次数有界，拥挤场地不会无限等待。
    max_batches = 200
    batch = max(2 * n_turbines, 10)
    attempts = 0
    while len(positions) < n_turbines and attempts < max_batches:
        attempts += 1
        try:
            candidates = boundary.sample_random_points(batch, rng, max_attempts=20)
        except RuntimeError:
            break
        placed = np.array(positions, dtype=np.float64) if positions else np.zeros((0, 2))
        for cand in candidates:
            if len(positions) >= n_turbines:
                break
            if constraint.candidate_acceptable(placed, cand):
                positions.append(cand)
                placed = np.array(positions, dtype=np.float64)

    if len(positions) < n_turbines:
        raise RuntimeError(
            f"场地内无法布置 {n_turbines} 台满足{constraint.describe()}的机组"
            f"（仅成功布置 {len(positions)} 台），场地可能过于拥挤"
        )

    positions = np.array(positions[:n_turbines], dtype=np.float64)

    # 有界修复：消除浮点误差或栅格/采样中残留的冲突；失败即抛出。
    valid, violations = constraint.check(positions)
    inside = boundary.contains_all(positions)
    if not (valid and bool(inside.all())):
        positions = enforce_spacing(
            positions, constraint, boundary, rng=rng, max_iterations=2000
        )

    return positions


def generate_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    aspect_ratio: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    spacing_constraint: Optional[SpacingConstraint] = None,
) -> np.ndarray:
    """生成规则网格布局作为优化基线。

    Parameters
    ----------
    boundary : SiteBoundary
        场地边界
    n_turbines : int
        风机台数
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        径向模式（未提供 ``spacing_constraint`` 时）的最小间距倍数
    aspect_ratio : float
        网格纵横比 (列数/行数)
    rng : Optional[np.random.Generator]
        随机数生成器
    spacing_constraint : Optional[SpacingConstraint]
        间距约束；给定时按其模式（径向/方向性椭圆）与参考方向铺设。

    Returns
    -------
    np.ndarray
        网格布局位置 (n_turbines, 2)，保证满足间距与边界约束
    """
    if rng is None:
        rng = np.random.default_rng()

    constraint = _build_constraint(rotor_diameters, min_multiple, spacing_constraint)
    return _generate_grid(
        boundary=boundary,
        n_turbines=n_turbines,
        rotor_diameters=rotor_diameters,
        constraint=constraint,
        staggered=False,
        aspect_ratio=aspect_ratio,
        rng=rng,
    )


def generate_staggered_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    dominant_direction: float = 270.0,
    rng: Optional[np.random.Generator] = None,
    spacing_constraint: Optional[SpacingConstraint] = None,
) -> np.ndarray:
    """生成交错网格布局（错位排列，减少主风向下的尾流）。

    方向性模式下列沿约束的参考风向铺设，奇数行沿顺风方向错开半个
    列距；径向模式下在世界坐标中错位（``dominant_direction`` 仅在
    未传入方向性约束时保留于接口，径向网格不旋转）。

    Returns
    -------
    np.ndarray
        交错网格布局位置 (n_turbines, 2)，保证满足间距与边界约束
    """
    if rng is None:
        rng = np.random.default_rng()

    constraint = _build_constraint(rotor_diameters, min_multiple, spacing_constraint)
    if spacing_constraint is None:
        # 保持旧接口语义：交错网格按主风向旋转（径向网格旋转后仍为
        # 等距方格，不影响合规性）。
        constraint = SpacingConstraint(
            mode="radial",
            rotor_diameters=rotor_diameters,
            min_multiple=min_multiple,
            reference_direction=dominant_direction,
        )

    return _generate_grid(
        boundary=boundary,
        n_turbines=n_turbines,
        rotor_diameters=rotor_diameters,
        constraint=constraint,
        staggered=True,
        aspect_ratio=1.0,
        rng=rng,
    )
