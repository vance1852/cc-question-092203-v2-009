"""风机间距约束。

支持两种安全间距口径：

* 径向（圆形）安全域：任意方向上机组间距不得小于 ``min_multiple * D``；
* 方向性（椭圆）安全域：在以参考风向为轴的旋转坐标系中，每对机组的
  顺风、横风分量必须落在椭圆
  ``(s_along / a)^2 + (s_cross / b)^2 >= 1`` 之外，其中
  ``a = downwind_multiple * D_ij``、``b = crosswind_multiple * D_ij``，
  ``D_ij`` 取两台机组转子直径的较大值。

边界点与零距离采用一致定义：引入统一的相对容差 ``tolerance``，归一化
比值 ``q >= 1 - tolerance`` 即视为满足（椭圆/圆周上的点可接受）；零距离时
``q = 0``，必然违规，所需净距按有限值报告，不会出现除零。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: 间距判定统一相对容差：q >= 1 - SPACING_TOLERANCE 即视为满足。
SPACING_TOLERANCE: float = 1e-9

#: 零距离判定阈值 (m)。
_ZERO_DISTANCE: float = 1e-12


class SpacingFeasibilityError(RuntimeError):
    """在给定场地内无法满足间距约束（有界退出）。"""


@dataclass
class SpacingViolation:
    """单机对的间距违规信息。

    Parameters
    ----------
    i, j : int
        违规机对的机组索引（i < j）。
    along_component : float
        机对位移在顺风轴上的有符号分量 (m)。
    cross_component : float
        机对位移在横风轴上的有符号分量 (m)。
    along_required : float
        顺风方向椭圆半轴（要求的顺风净距尺度）(m)。
    cross_required : float
        横风方向椭圆半轴（要求的横风净距尺度）(m)。
    distance : float
        机对欧氏距离 (m)。
    required_clearance : float
        沿机对当前连线方向仍需拉开的净距 (m)；零距离时报告最小半轴，
        保证为有限值。
    normalized_ratio : float
        归一化比值 q，q < 1 表示落在安全域内部。
    """

    i: int
    j: int
    along_component: float
    cross_component: float
    along_required: float
    cross_required: float
    distance: float
    required_clearance: float
    normalized_ratio: float


class SpacingConstraint:
    """机组间距约束（径向圆域或方向性椭圆域）。

    Parameters
    ----------
    rotor_diameters : np.ndarray
        每台机组的转子直径 (m)，形状 (N,)。
    directional : bool
        是否启用方向性间距。False 时退化为各向同性的径向圆域。
    reference_direction : float
        参考风向（度，气象惯例：0 度为正北、顺时针），顺风轴沿该风的
        流动方向。例如 270 度表示西风，顺风轴指向正东。
    downwind_multiple : float
        顺风安全间距倍数（相对转子直径）。
    crosswind_multiple : float
        横风安全间距倍数（相对转子直径）。
    min_spacing_multiple : float
        径向模式（``directional=False``）下的最小间距倍数。
    tolerance : float
        归一化边界相对容差。
    """

    def __init__(
        self,
        rotor_diameters: np.ndarray,
        *,
        directional: bool = False,
        reference_direction: float = 270.0,
        downwind_multiple: float = 7.0,
        crosswind_multiple: float = 3.0,
        min_spacing_multiple: float = 5.0,
        tolerance: float = SPACING_TOLERANCE,
    ) -> None:
        diameters = np.asarray(rotor_diameters, dtype=np.float64).reshape(-1)
        if diameters.size == 0:
            raise ValueError("至少需要一台机组的转子直径")
        if np.any(diameters <= 0.0):
            raise ValueError("转子直径必须为正数")
        if downwind_multiple <= 0.0 or crosswind_multiple <= 0.0:
            raise ValueError("顺风/横风间距倍数必须为正数")
        if min_spacing_multiple <= 0.0:
            raise ValueError("最小间距倍数必须为正数")
        if not 0.0 <= tolerance < 1.0:
            raise ValueError("容差必须位于 [0, 1) 区间")

        self.rotor_diameters = diameters
        self.n_turbines = diameters.size
        self.directional = bool(directional)
        self.reference_direction = float(reference_direction)
        self.downwind_multiple = float(downwind_multiple)
        self.crosswind_multiple = float(crosswind_multiple)
        self.min_spacing_multiple = float(min_spacing_multiple)
        self.tolerance = float(tolerance)

        # 与全项目风向约定一致：wind_rad = deg2rad(270 - direction)。
        wind_rad = np.deg2rad(270.0 - self.reference_direction)
        self.e_along = np.array([np.cos(wind_rad), np.sin(wind_rad)], dtype=np.float64)
        # 横风轴：顺风轴逆时针旋转 90 度。
        self.e_cross = np.array(
            [-np.sin(wind_rad), np.cos(wind_rad)], dtype=np.float64
        )

    # ---- 基本量 -------------------------------------------------

    def pair_diameter(self, i: int, j: int) -> float:
        """机对特征转子直径：取两台直径的较大值。"""
        return float(max(self.rotor_diameters[i], self.rotor_diameters[j]))

    def pair_semiaxes(self, i: int, j: int) -> tuple[float, float]:
        """返回机对要求的（顺风半轴, 横风半轴）(m)。"""
        d_ij = self.pair_diameter(i, j)
        if self.directional:
            return self.downwind_multiple * d_ij, self.crosswind_multiple * d_ij
        radius = self.min_spacing_multiple * d_ij
        return radius, radius

    def project(self, delta: np.ndarray) -> tuple[float, float]:
        """将位移向量投影到旋转坐标系，返回有符号的（顺风, 横风）分量。"""
        delta = np.asarray(delta, dtype=np.float64)
        return float(delta @ self.e_along), float(delta @ self.e_cross)

    @property
    def characteristic_diameter(self) -> float:
        """代表性安全域使用的特征直径（全场最大转子直径）。"""
        return float(np.max(self.rotor_diameters))

    def representative_semiaxes(self) -> tuple[float, float]:
        """代表性安全域的（顺风半轴, 横风半轴）(m)。"""
        d_max = self.characteristic_diameter
        if self.directional:
            return self.downwind_multiple * d_max, self.crosswind_multiple * d_max
        radius = self.min_spacing_multiple * d_max
        return radius, radius

    @property
    def ellipse_angle_deg(self) -> float:
        """椭圆长轴相对 x 轴逆时针的角度（度），供绘图使用。"""
        return float(np.rad2deg(np.arctan2(self.e_along[1], self.e_along[0])))

    def describe(self) -> str:
        """人类可读的约束描述。"""
        if self.directional:
            return (
                f"方向性椭圆间距（参考风向 {self.reference_direction:.0f}°，"
                f"顺风 {self.downwind_multiple:g}D、横风 {self.crosswind_multiple:g}D）"
            )
        return f"径向最小间距 {self.min_spacing_multiple:g}D"

    # ---- 约束判定 -----------------------------------------------

    def evaluate_pair(
        self,
        positions: np.ndarray,
        i: int,
        j: int,
    ) -> Optional[SpacingViolation]:
        """评估单个机对；满足约束返回 None，否则返回违规详情。"""
        positions = np.asarray(positions, dtype=np.float64)
        delta = positions[j] - positions[i]
        s_along, s_cross = self.project(delta)
        a, b = self.pair_semiaxes(i, j)

        distance = float(np.hypot(delta[0], delta[1]))
        q = float(np.sqrt((s_along / a) ** 2 + (s_cross / b) ** 2))

        if q >= 1.0 - self.tolerance:
            return None

        if distance < _ZERO_DISTANCE:
            # 零距离时连线方向未定义，报告离开椭圆所需的最小净距。
            required_clearance = float(min(a, b))
        else:
            u_along = s_along / distance
            u_cross = s_cross / distance
            radius_in_pair_direction = 1.0 / np.sqrt(
                (u_along / a) ** 2 + (u_cross / b) ** 2
            )
            required_clearance = float(
                (1.0 - q) * radius_in_pair_direction
            )

        return SpacingViolation(
            i=i,
            j=j,
            along_component=s_along,
            cross_component=s_cross,
            along_required=float(a),
            cross_required=float(b),
            distance=distance,
            required_clearance=required_clearance,
            normalized_ratio=q,
        )

    def check(self, positions: np.ndarray) -> tuple[bool, list[SpacingViolation]]:
        """检查所有机对。

        Returns
        -------
        tuple[bool, list[SpacingViolation]]
            是否全部满足，以及违规机对详情列表（含方向分量与所需净距）。
        """
        positions = np.asarray(positions, dtype=np.float64)
        violations: list[SpacingViolation] = []
        for i in range(self.n_turbines):
            for j in range(i + 1, self.n_turbines):
                violation = self.evaluate_pair(positions, i, j)
                if violation is not None:
                    violations.append(violation)
        return (not violations), violations

    def is_candidate_feasible(
        self,
        positions: np.ndarray,
        candidate: np.ndarray,
        candidate_idx: Optional[int] = None,
    ) -> bool:
        """判断候选机位相对已布机位是否满足间距约束（用于拒绝采样）。

        Parameters
        ----------
        positions : np.ndarray
            已布置的机组位置，形状 (K, 2)。
        candidate : np.ndarray
            候选机位，形状 (2,)。
        candidate_idx : Optional[int]
            候选机对应的机组索引（决定其转子直径）；默认为第 K 台。
        """
        positions = np.asarray(positions, dtype=np.float64)
        candidate = np.asarray(candidate, dtype=np.float64)
        if candidate_idx is None:
            candidate_idx = positions.shape[0]
        d_j = float(self.rotor_diameters[candidate_idx])

        for i in range(positions.shape[0]):
            delta = candidate - positions[i]
            d_ij = max(float(self.rotor_diameters[i]), d_j)
            if self.directional:
                a = self.downwind_multiple * d_ij
                b = self.crosswind_multiple * d_ij
            else:
                a = b = self.min_spacing_multiple * d_ij

            s_along, s_cross = self.project(delta)
            q = float(np.hypot(s_along / a, s_cross / b))
            if q < 1.0 - self.tolerance:
                return False
        return True

    # ---- 修复 ---------------------------------------------------

    def enforce(
        self,
        positions: np.ndarray,
        boundary,
        rng: Optional[np.random.Generator] = None,
        max_iterations: int = 1000,
    ) -> np.ndarray:
        """尝试通过推开机组满足间距约束，同时保持机组位于场地内。

        每轮迭代把违规机对沿其连线方向各推开所需净距的一半；零距离机对
        沿随机方向分离。无法在 ``max_iterations`` 内收敛时抛出
        :class:`SpacingFeasibilityError`（有界退出）。
        """
        if rng is None:
            rng = np.random.default_rng()

        positions = np.array(positions, dtype=np.float64, copy=True)

        for _ in range(max_iterations):
            valid, violations = self.check(positions)
            if valid:
                return positions

            for v in violations:
                i, j = v.i, v.j
                vec = positions[j] - positions[i]
                dist = float(np.linalg.norm(vec))
                if dist < _ZERO_DISTANCE:
                    direction = rng.standard_normal(2)
                    direction /= np.linalg.norm(direction)
                else:
                    direction = vec / dist

                a, b = self.pair_semiaxes(i, j)
                if dist < _ZERO_DISTANCE:
                    u_along = float(direction @ self.e_along)
                    u_cross = float(direction @ self.e_cross)
                    radius_in_direction = 1.0 / np.sqrt(
                        (u_along / a) ** 2 + (u_cross / b) ** 2
                    )
                    push = radius_in_direction / 2.0 + 1e-6
                else:
                    push = v.required_clearance / 2.0 + 1e-6

                positions[i] -= direction * push
                positions[j] += direction * push

            for k in range(self.n_turbines):
                if not boundary.contains_point(positions[k]):
                    positions[k] = boundary.project_to_boundary(positions[k])
                    perturbation = rng.uniform(-5.0, 5.0, 2)
                    positions[k] += perturbation
                    if not boundary.contains_point(positions[k]):
                        positions[k] = boundary.project_to_boundary(positions[k])

        valid, _ = self.check(positions)
        inside = boundary.contains_all(positions)
        if not (valid and inside.all()):
            raise SpacingFeasibilityError("无法通过调整满足间距和边界约束")

        return positions


# ----------------------------------------------------------------------
# 径向模式的兼容接口（关闭方向性规则时继续使用）
# ----------------------------------------------------------------------


def check_min_spacing(
    positions: np.ndarray,
    min_distance: float,
) -> tuple[bool, np.ndarray]:
    """检查所有风机对之间的间距是否满足最小距离要求。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N_turbines, 2)
    min_distance : float
        最小允许间距 (m)

    Returns
    -------
    tuple[bool, np.ndarray]
        - 是否所有间距都满足要求
        - 不满足要求的风机对索引数组，形状为 (M, 2)，M 为违规对数
    """
    n = positions.shape[0]
    violations = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < min_distance:
                violations.append([i, j])

    if violations:
        return False, np.array(violations, dtype=int)
    else:
        return True, np.zeros((0, 2), dtype=int)


def check_directional_spacing(
    positions: np.ndarray,
    rotor_diameters: np.ndarray,
    reference_direction: float = 270.0,
    downwind_multiple: float = 7.0,
    crosswind_multiple: float = 3.0,
    tolerance: float = SPACING_TOLERANCE,
) -> tuple[bool, list[SpacingViolation]]:
    """方向性椭圆间距检查的便捷函数。

    Returns
    -------
    tuple[bool, list[SpacingViolation]]
        是否全部满足，以及违规机对详情（方向分量、所需净距）。
    """
    constraint = SpacingConstraint(
        rotor_diameters,
        directional=True,
        reference_direction=reference_direction,
        downwind_multiple=downwind_multiple,
        crosswind_multiple=crosswind_multiple,
        tolerance=tolerance,
    )
    return constraint.check(positions)


def compute_min_spacing_from_diameters(
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
) -> float:
    """根据转子直径计算最小间距（取最大直径的倍数）。

    Parameters
    ----------
    rotor_diameters : np.ndarray
        每台风机的转子直径
    min_multiple : float
        最小间距倍数（相对于转子直径）

    Returns
    -------
    float
        最小间距 (m)
    """
    return float(min_multiple * np.max(rotor_diameters))


def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """计算所有风机对之间的距离矩阵。

    Parameters
    ----------
    positions : np.ndarray
        风机位置，形状为 (N, 2)

    Returns
    -------
    np.ndarray
        距离矩阵，形状为 (N, N)，对角线为 0
    """
    n = positions.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def enforce_min_spacing(
    positions: np.ndarray,
    min_distance: float,
    boundary,
    rng: np.random.Generator | None = None,
    max_iterations: int = 1000,
) -> np.ndarray:
    """尝试通过移动风机来满足径向最小间距约束（兼容接口）。

    内部基于 :class:`SpacingConstraint` 的径向模式实现。
    """
    if rng is None:
        rng = np.random.default_rng()

    n = positions.shape[0]
    diameters = np.full(n, min_distance / 5.0, dtype=np.float64)
    constraint = SpacingConstraint(
        diameters,
        directional=False,
        min_spacing_multiple=5.0,
    )
    return constraint.enforce(
        positions, boundary, rng=rng, max_iterations=max_iterations
    )
