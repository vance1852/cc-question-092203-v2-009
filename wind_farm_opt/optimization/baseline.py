"""基线布局生成（规则网格、交错网格）。

网格在间距约束给定的旋转坐标系中铺设：列沿顺风轴、行沿横风轴，列/行
间距分别取顺风、横风椭圆半轴（留少量数值余量），因此铺设出的相邻机组
天然落在椭圆安全域之外。径向模式下两轴等长且不旋转（默认参考风向
270° 时顺风轴与 x 轴重合），退化为原有正方形网格。
"""

import numpy as np

from ..constraints.boundary import SiteBoundary
from ..constraints.spacing import (
    SpacingConstraint,
    SpacingFeasibilityError,
)

#: 网格步长相对安全半轴的放宽上限（沿用历史径向网格的 1.5 倍口径）。
_GRID_SPREAD = 1.5
#: 拒绝采样补足机位时的总尝试上限。
_FILL_MAX_ATTEMPTS = 5000


def _resolve_constraint(
    rotor_diameters: np.ndarray,
    min_multiple: float,
    spacing_constraint: SpacingConstraint | None,
) -> SpacingConstraint:
    """未显式传入约束时，按径向倍数构造兼容的径向约束。"""
    if spacing_constraint is not None:
        return spacing_constraint
    return SpacingConstraint(
        rotor_diameters,
        directional=False,
        min_spacing_multiple=min_multiple,
    )


def _projected_bounds(
    boundary: SiteBoundary,
    e_along: np.ndarray,
    e_cross: np.ndarray,
) -> tuple[float, float, float, float]:
    """场地包围盒角点在顺风/横风轴上的投影范围。"""
    corners = np.array([
        [boundary.x_min, boundary.y_min],
        [boundary.x_max, boundary.y_min],
        [boundary.x_max, boundary.y_max],
        [boundary.x_min, boundary.y_max],
    ])
    along = corners @ e_along
    cross = corners @ e_cross
    return float(along.min()), float(along.max()), float(cross.min()), float(cross.max())


def _fill_rejection(
    positions: list[np.ndarray],
    n_turbines: int,
    boundary: SiteBoundary,
    constraint: SpacingConstraint,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """用拒绝采样补足机位，候选点必须同时落在场内和所有安全域外。"""
    candidate_budget = _FILL_MAX_ATTEMPTS
    while len(positions) < n_turbines and candidate_budget > 0:
        try:
            candidates = boundary.sample_random_points(
                min(n_turbines * 2, 40), rng, max_attempts=20
            )
        except RuntimeError as exc:
            raise SpacingFeasibilityError(
                f"场地内无法布置 {n_turbines} 台满足间距约束的机组"
                f"（仅成功布置 {len(positions)} 台）"
            ) from exc

        candidate_budget -= len(candidates)
        for cand in candidates:
            if len(positions) >= n_turbines:
                break
            existing = np.array(positions, dtype=np.float64)
            if constraint.is_candidate_feasible(existing, cand):
                positions.append(cand)

    if len(positions) < n_turbines:
        raise SpacingFeasibilityError(
            f"场地内无法布置 {n_turbines} 台满足间距约束的机组"
            f"（仅成功布置 {len(positions)} 台）"
        )
    return positions


def _finalize_layout(
    positions: list[np.ndarray],
    boundary: SiteBoundary,
    constraint: SpacingConstraint,
    rng: np.random.Generator,
) -> np.ndarray:
    """校验布局；不合规则尝试修复，仍失败则有界退出。"""
    positions = np.array(positions, dtype=np.float64)

    valid, violations = constraint.check(positions)
    inside = boundary.contains_all(positions).all()

    if not (valid and inside):
        positions = constraint.enforce(positions, boundary, rng)

    # enforce 成功后必然合规；这里再做一次断言式校验。
    valid, violations = constraint.check(positions)
    if not valid or not boundary.contains_all(positions).all():
        raise SpacingFeasibilityError("生成的基线布局不满足间距约束")

    return positions


def _generate_grid(
    boundary: SiteBoundary,
    n_turbines: int,
    constraint: SpacingConstraint,
    rng: np.random.Generator,
    staggered: bool,
    aspect_ratio: float = 1.0,
) -> np.ndarray:
    """在旋转坐标系中铺设网格（可选交错）。"""
    a, b = constraint.representative_semiaxes()

    # 行列数沿用历史口径：规则网格 round/ceil，交错网格 floor(sqrt)/ceil。
    if staggered:
        n_rows = max(1, int(np.sqrt(n_turbines)))
    else:
        n_rows = max(1, int(np.round(np.sqrt(n_turbines / aspect_ratio))))
    n_cols = max(1, int(np.ceil(n_turbines / n_rows)))

    along_min, along_max, cross_min, cross_max = _projected_bounds(
        boundary, constraint.e_along, constraint.e_cross
    )

    # 两种历史铺设口径推广到旋转坐标系：
    # 规则网格取 min(跨度/(格数-1), 1.5*半轴)，并钳制到不小于半轴；
    # 交错网格取 max(跨度/(格数-1), 1.2*半轴)。
    # 超出场地的格点由边界过滤剔除，再用拒绝采样补足机位。
    margin_along = a * 0.5
    margin_cross = b * 0.5
    along_span = (along_max - along_min) - 2.0 * margin_along
    cross_span = (cross_max - cross_min) - 2.0 * margin_cross

    if staggered:
        step_along = max(along_span / max(n_cols - 1, 1), a * 1.2)
        step_cross = max(cross_span / max(n_rows - 1, 1), b * 1.2)
    else:
        step_along = max(
            min(along_span / max(n_cols - 1, 1), a * _GRID_SPREAD), a
        )
        step_cross = max(
            min(cross_span / max(n_rows - 1, 1), b * _GRID_SPREAD), b
        )

    used_along = step_along * (n_cols - 1)
    used_cross = step_cross * (n_rows - 1)
    start_along = along_min + margin_along + (along_span - used_along) / 2.0
    start_cross = cross_min + margin_cross + (cross_span - used_cross) / 2.0

    positions: list[np.ndarray] = []
    for row in range(n_rows):
        offset = step_along / 2.0 if (staggered and row % 2 == 1) else 0.0
        for col in range(n_cols):
            if len(positions) >= n_turbines:
                break
            u = start_along + col * step_along + offset
            v = start_cross + row * step_cross
            pos = u * constraint.e_along + v * constraint.e_cross
            if boundary.contains_point(pos):
                positions.append(pos)

    positions = _fill_rejection(positions, n_turbines, boundary, constraint, rng)
    return _finalize_layout(positions, boundary, constraint, rng)


def generate_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    aspect_ratio: float = 1.0,
    rng: np.random.Generator | None = None,
    spacing_constraint: SpacingConstraint | None = None,
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
        径向模式下的最小间距倍数（``spacing_constraint`` 为空时使用）
    aspect_ratio : float
        网格纵横比 (列数/行数)
    rng : Optional[np.random.Generator]
        随机数生成器
    spacing_constraint : Optional[SpacingConstraint]
        间距约束；传入方向性约束时网格按参考风向旋转铺设

    Returns
    -------
    np.ndarray
        网格布局位置 (n_turbines, 2)，保证满足间距与边界约束；
        场地无法容纳时抛出 :class:`SpacingFeasibilityError`
    """
    if rng is None:
        rng = np.random.default_rng()

    constraint = _resolve_constraint(rotor_diameters, min_multiple, spacing_constraint)
    return _generate_grid(
        boundary, n_turbines, constraint, rng,
        staggered=False, aspect_ratio=aspect_ratio,
    )


def generate_staggered_grid_layout(
    boundary: SiteBoundary,
    n_turbines: int,
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
    dominant_direction: float = 270.0,
    rng: np.random.Generator | None = None,
    spacing_constraint: SpacingConstraint | None = None,
) -> np.ndarray:
    """生成交错网格布局（奇数行错位半个顺风列距）。

    传入方向性约束时，交错方向沿顺风轴、行距沿横风轴，并整体按参考风向
    旋转；未传入时按 ``dominant_direction`` 构造旋转坐标系（若与默认
    270° 不同则网格旋转），径向倍数仍取 ``min_multiple``。

    Returns
    -------
    np.ndarray
        交错网格布局位置 (n_turbines, 2)，保证满足约束；
        场地无法容纳时抛出 :class:`SpacingFeasibilityError`
    """
    if rng is None:
        rng = np.random.default_rng()

    if spacing_constraint is not None:
        constraint = spacing_constraint
    else:
        constraint = SpacingConstraint(
            rotor_diameters,
            directional=False,
            reference_direction=dominant_direction,
            min_spacing_multiple=min_multiple,
        )

    return _generate_grid(
        boundary, n_turbines, constraint, rng, staggered=True
    )
