"""风机间距约束。

支持两种安全间距模式：

- 径向模式（传统）：所有方向使用同一个最小圆形距离 ``min_multiple * D``；
- 方向性模式（审批椭圆安全域）：以审批给定的参考风向为长轴方向，
  顺风/横风分别使用不同倍数。机组对 (i, j) 在随参考方向旋转后的
  坐标系 ``(along, cross)`` 中必须位于彼此的椭圆安全域之外：

      (along / a_ij)^2 + (cross / b_ij)^2 >= 1

  其中 ``a_ij = downwind_multiple * max(D_i, D_j)``、
  ``b_ij = crosswind_multiple * max(D_i, D_j)``。两台风机各自的椭圆
  域同方向、按直径等比缩放，嵌套时由较大直径的椭圆起控制作用，因此
  机对判定取两者直径的最大值（仍逐对依据自身直径计算）。

数值约定（两种模式一致）：

- ``SPACING_TOLERANCE``：无量纲相对容差。恰好在安全域边界上的机对
  判为合格；径向下等价于 ``min_distance * (1 - TOL)`` 的物理容差。
- ``ZERO_DISTANCE``：米。机对距离不大于该值时视为重合（零距离），
  任何非零安全域下都判为违规；修复时按重合对处理（随机方向推开），
  从而避免除零，且零距离不会被任何容差“放行”。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# 无量纲相对容差：边界点（椭圆 q≈1 或径向 r≈min_distance）判为合格。
SPACING_TOLERANCE = 1e-9
# 重合（零距离）阈值 (m)：低于该距离按重合处理，始终违规、禁止除零。
ZERO_DISTANCE = 1e-12


@dataclass
class PairViolation:
    """一对违反间距约束的机组及其方向信息。

    Attributes
    ----------
    i, j : int
        机组索引。
    along : float
        机对连线在参考风向上的有向分量 (m)，``p_j - p_i`` 投影到
        顺风方向。径向模式下同样按参考方向给出，便于报告。
    cross : float
        机对连线在横风方向上的有向分量 (m)。
    distance : float
        机对中心欧氏距离 (m)；重合时为 0。
    required_along : float
        顺风方向所需净距 (m)：方向性模式为顺风半轴长 a，
        径向模式为圆形最小距离。
    required_cross : float
        横风方向所需净距 (m)：方向性模式为横风半轴长 b，
        径向模式为圆形最小距离。
    required_distance : float
        沿机对当前方位离开安全域所需的中心欧氏净距 (m)。
        重合（方位无定义）时取长半轴 a，保证任意方向推开均可出域。
    binding_diameter : float
        起控制作用的转子直径 (m)（机对两者的较大值）。
    """

    i: int
    j: int
    along: float
    cross: float
    distance: float
    required_along: float
    required_cross: float
    required_distance: float
    binding_diameter: float

    @property
    def pair(self) -> tuple[int, int]:
        return (self.i, self.j)

    @property
    def shortfall(self) -> float:
        """所需净距与当前距离之差 (m)，重合时即所需净距本身。"""
        return max(0.0, self.required_distance - self.distance)


class SpacingConstraint:
    """机组间距约束（径向圆形或方向性椭圆）。

    Parameters
    ----------
    mode : str
        ``"radial"``（径向）或 ``"directional"``（方向性椭圆）。
    rotor_diameters : np.ndarray
        每台机组的转子直径 (m)，形状 (N,)。
    min_multiple : float
        径向模式下的最小间距倍数（相对转子直径）。
    reference_direction : float
        方向性模式下的参考风向（度，气象习惯：风的来向），
        与风资源/尾流模块使用同一约定。
    downwind_multiple, crosswind_multiple : float
        顺风、横风方向的间距倍数（相对转子直径）。
    min_distance : float, optional
        直接给定径向最小距离 (m)，给定时忽略直径与倍数。
    tolerance : float
        无量纲相对容差，见模块说明。
    """

    def __init__(
        self,
        mode: str = "radial",
        rotor_diameters: Optional[np.ndarray] = None,
        min_multiple: float = 5.0,
        reference_direction: float = 270.0,
        downwind_multiple: float = 7.0,
        crosswind_multiple: float = 3.0,
        min_distance: Optional[float] = None,
        tolerance: float = SPACING_TOLERANCE,
    ) -> None:
        mode = mode.lower()
        if mode not in ("radial", "directional"):
            raise ValueError(f"未知的间距模式: {mode!r}，应为 radial 或 directional")
        self.mode = mode

        if rotor_diameters is None:
            rotor_diameters = np.ones(1, dtype=np.float64)
        self.rotor_diameters = np.asarray(rotor_diameters, dtype=np.float64)
        if self.rotor_diameters.ndim != 1 or np.any(self.rotor_diameters <= 0.0):
            raise ValueError("转子直径必须为正数的一维数组")

        if min_multiple <= 0.0 or downwind_multiple <= 0.0 or crosswind_multiple <= 0.0:
            raise ValueError("间距倍数必须为正数")
        if not 0.0 <= tolerance < 1.0:
            raise ValueError("容差必须在 [0, 1) 范围内")

        self.min_multiple = float(min_multiple)
        self.reference_direction = float(reference_direction)
        self.downwind_multiple = float(downwind_multiple)
        self.crosswind_multiple = float(crosswind_multiple)
        self.tolerance = float(tolerance)
        self._min_distance_override = (
            float(min_distance) if min_distance is not None else None
        )

        # 气流前进方向单位向量。气象角度 -> 世界坐标的换算与
        # wind_resource / wake 热力图保持一致：theta = 270 - direction。
        theta = np.deg2rad(270.0 - self.reference_direction)
        self._e_along = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        self._e_cross = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)

    # ---- 构造工具 ---------------------------------------------------

    @classmethod
    def radial(
        cls,
        rotor_diameters: np.ndarray,
        min_multiple: float = 5.0,
        min_distance: Optional[float] = None,
    ) -> "SpacingConstraint":
        """创建径向（圆形）间距约束。"""
        return cls(
            mode="radial",
            rotor_diameters=rotor_diameters,
            min_multiple=min_multiple,
            min_distance=min_distance,
        )

    @classmethod
    def directional(
        cls,
        rotor_diameters: np.ndarray,
        reference_direction: float = 270.0,
        downwind_multiple: float = 7.0,
        crosswind_multiple: float = 3.0,
    ) -> "SpacingConstraint":
        """创建方向性（椭圆）间距约束。"""
        return cls(
            mode="directional",
            rotor_diameters=rotor_diameters,
            reference_direction=reference_direction,
            downwind_multiple=downwind_multiple,
            crosswind_multiple=crosswind_multiple,
        )

    # ---- 基本量 -----------------------------------------------------

    @property
    def is_directional(self) -> bool:
        """是否为方向性椭圆模式（关闭新规则时为 False，使用径向间距）。"""
        return self.mode == "directional"

    @property
    def e_along(self) -> np.ndarray:
        """顺风方向单位向量（世界坐标），形状 (2,)。"""
        return self._e_along.copy()

    @property
    def e_cross(self) -> np.ndarray:
        """横风方向单位向量（世界坐标），形状 (2,)。"""
        return self._e_cross.copy()

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """返回旋转坐标系的两个单位轴 (顺风, 横风)。"""
        return self._e_along.copy(), self._e_cross.copy()

    @property
    def d_max(self) -> float:
        return float(np.max(self.rotor_diameters))

    @property
    def radial_min_distance(self) -> float:
        """径向模式下的最小圆形距离 (m)。"""
        if self._min_distance_override is not None:
            return self._min_distance_override
        return self.min_multiple * self.d_max

    def semiaxes(self, diameter: float) -> tuple[float, float]:
        """给定转子直径对应的 (顺风半轴 a, 横风半轴 b)，单位 m。

        径向模式下两者相等，均为圆形最小距离。
        """
        if self.is_directional:
            return (
                self.downwind_multiple * float(diameter),
                self.crosswind_multiple * float(diameter),
            )
        r = self.radial_min_distance
        return r, r

    def pair_semiaxes(self, i: int, j: int) -> tuple[float, float, float]:
        """机对 (i, j) 的控制直径与安全域半轴。

        Returns
        -------
        tuple[float, float, float]
            (binding_diameter, a, b)：机对两者较大转子直径及对应的
            顺风/横风半轴（径向模式 a == b == 最小距离）。
        """
        d = float(max(self.rotor_diameters[i], self.rotor_diameters[j]))
        a, b = self.semiaxes(d)
        return d, a, b

    def components(
        self, pos_i: np.ndarray, pos_j: np.ndarray
    ) -> tuple[float, float, float]:
        """机对向量 ``p_j - p_i`` 的 (顺风分量, 横风分量, 欧氏距离)。"""
        delta = np.asarray(pos_j, dtype=np.float64) - np.asarray(
            pos_i, dtype=np.float64
        )
        along = float(delta @ self._e_along)
        cross = float(delta @ self._e_cross)
        dist = float(np.hypot(along, cross))
        return along, cross, dist

    # ---- 约束判定 ---------------------------------------------------

    def evaluate_pair(
        self,
        pos_i: np.ndarray,
        pos_j: np.ndarray,
        diameter_i: float,
        diameter_j: float,
    ) -> dict:
        """评估单个机对，返回方向分量与所需净距的详细信息。

        两台机组的椭圆安全域同向、按各自直径等比缩放，两域互相
        嵌套，故仅需用较大直径（``binding_diameter``）的椭圆判定
        即可，它对两台机组同时构成最严格约束。
        """
        along, cross, dist = self.components(pos_i, pos_j)
        d_bind = float(max(diameter_i, diameter_j))
        a, b = self.semiaxes(d_bind)

        if self.is_directional:
            required_along, required_cross = a, b
            if dist <= ZERO_DISTANCE:
                # 零距离：方位无定义，q=0；任意方向推开至长半轴外才保险。
                q = 0.0
                required_distance = a
            else:
                q = (along / a) ** 2 + (cross / b) ** 2
                # 沿当前方位的出域距离：1 / sqrt((u_hat/a)^2 + (v_hat/b)^2)
                required_distance = 1.0 / np.sqrt(
                    (along / (dist * a)) ** 2 + (cross / (dist * b)) ** 2
                )
            violates = dist <= ZERO_DISTANCE or q < 1.0 - self.tolerance
        else:
            required_along = required_cross = self.radial_min_distance
            required_distance = self.radial_min_distance
            q = None
            violates = dist + self.tolerance * self.radial_min_distance < required_distance

        return {
            "along": along,
            "cross": cross,
            "distance": dist,
            "ellipse_value": q,
            "required_along": required_along,
            "required_cross": required_cross,
            "required_distance": float(required_distance),
            "binding_diameter": d_bind,
            "violates": bool(violates),
        }

    def check(self, positions: np.ndarray) -> tuple[bool, list[PairViolation]]:
        """检查全部机对是否满足间距约束。

        Returns
        -------
        tuple[bool, list[PairViolation]]
            是否全部合格，以及违规机对列表（含方向分量与所需净距）。
        """
        positions = np.asarray(positions, dtype=np.float64)
        n = positions.shape[0]
        violations: list[PairViolation] = []

        for i in range(n):
            for j in range(i + 1, n):
                info = self.evaluate_pair(
                    positions[i],
                    positions[j],
                    self.rotor_diameters[i],
                    self.rotor_diameters[j],
                )
                if info["violates"]:
                    violations.append(
                        PairViolation(
                            i=i,
                            j=j,
                            along=info["along"],
                            cross=info["cross"],
                            distance=info["distance"],
                            required_along=info["required_along"],
                            required_cross=info["required_cross"],
                            required_distance=info["required_distance"],
                            binding_diameter=info["binding_diameter"],
                        )
                    )

        return (len(violations) == 0), violations

    def is_satisfied(self, positions: np.ndarray) -> bool:
        """布局是否满足间距约束（布尔便捷接口）。"""
        valid, _ = self.check(positions)
        return valid

    def candidate_acceptable(
        self, placed: np.ndarray, candidate: np.ndarray, candidate_index: Optional[int] = None
    ) -> bool:
        """判断候选机位相对于已布机组是否满足约束（拒绝采样用）。

        Parameters
        ----------
        placed : np.ndarray
            已布机组位置 (M, 2)。
        candidate : np.ndarray
            候选机位 (2,)。
        candidate_index : int, optional
            候选机组在完整机组列表中的索引（用于取其转子直径）；
            缺省时按所有机组的最大直径处理（偏保守）。
        """
        d_cand = (
            self.rotor_diameters[candidate_index]
            if candidate_index is not None
            else self.d_max
        )
        for k in range(placed.shape[0]):
            info = self.evaluate_pair(
                placed[k], candidate, self.rotor_diameters[k], d_cand
            )
            if info["violates"]:
                return False
        return True

    def describe(self) -> str:
        """人类可读的约束描述（日志/打印用）。"""
        if self.is_directional:
            d = self.d_max
            a, b = self.semiaxes(d)
            return (
                f"方向性椭圆间距(参考风向 {self.reference_direction:.0f}°, "
                f"顺风 {self.downwind_multiple:g}D={a:.1f} m, "
                f"横风 {self.crosswind_multiple:g}D={b:.1f} m)"
            )
        return f"径向圆形间距({self.min_multiple:g}D={self.radial_min_distance:.1f} m)"


# --------------------------------------------------------------------
# 向后兼容的函数接口（内部统一委托给 SpacingConstraint）
# --------------------------------------------------------------------


def check_min_spacing(
    positions: np.ndarray,
    min_distance: float,
) -> tuple[bool, np.ndarray]:
    """检查所有风机对之间的间距是否满足最小圆形距离要求。

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
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    violations = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            if dist <= ZERO_DISTANCE or dist + SPACING_TOLERANCE * min_distance < min_distance:
                violations.append([i, j])

    if violations:
        return False, np.array(violations, dtype=int)
    return True, np.zeros((0, 2), dtype=int)


def compute_min_spacing_from_diameters(
    rotor_diameters: np.ndarray,
    min_multiple: float = 5.0,
) -> float:
    """根据转子直径计算最小间距（取最大直径的倍数）。"""
    return float(min_multiple * np.max(rotor_diameters))


def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """计算所有风机对之间的距离矩阵，对角线为 0。"""
    n = positions.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            dist[i, j] = d
            dist[j, i] = d
    return dist


def enforce_spacing(
    positions: np.ndarray,
    constraint: SpacingConstraint,
    boundary,
    rng: Optional[np.random.Generator] = None,
    max_iterations: int = 1000,
) -> np.ndarray:
    """尝试通过移动机组使布局满足间距约束与边界约束（有界修复）。

    径向模式下将违规机对沿连线方向对半推开；方向性模式下沿椭圆
    安全域在该点的外法向（旋转坐标系中 ``(along/a^2, cross/b^2)``
    变换回世界坐标）推开，推开量为当前方位上的出域净距缺口。

    重合对（零距离）沿随机方向推开至长半轴之外，杜绝除零。

    迭代次数受 ``max_iterations`` 限制；收敛后若仍不满足约束或
    机组位于场外，则抛出 :class:`RuntimeError`（拥挤场地下的有界
    退出），不会无限循环，也不会返回违规布局。
    """
    if rng is None:
        rng = np.random.default_rng()

    positions = np.array(positions, dtype=np.float64, copy=True)
    n = positions.shape[0]
    push_eps = 1e-6  # m，出域后留微小物理余量（容差用于判定，此量用于收敛）

    for _ in range(max_iterations):
        valid, violations = constraint.check(positions)
        inside = boundary.contains_all(positions)
        if valid and bool(inside.all()):
            return positions

        for v in violations:
            i, j = v.i, v.j
            if constraint.is_directional and v.distance > ZERO_DISTANCE:
                # 椭圆 q=(along/a)^2+(cross/b)^2 的外法向。
                _, a, b = constraint.pair_semiaxes(i, j)
                n_rot = np.array(
                    [v.along / a**2, v.cross / b**2], dtype=np.float64
                )
                n_norm = float(np.linalg.norm(n_rot))
                if n_norm <= ZERO_DISTANCE:
                    direction = rng.standard_normal(2)
                    direction /= np.linalg.norm(direction)
                else:
                    e_along, e_cross = constraint.axes()
                    n_rot /= n_norm
                    direction = n_rot[0] * e_along + n_rot[1] * e_cross
            else:
                vec = positions[j] - positions[i]
                dist = float(np.linalg.norm(vec))
                if dist <= ZERO_DISTANCE:
                    direction = rng.standard_normal(2)
                    direction /= np.linalg.norm(direction)
                else:
                    direction = vec / dist

            shortfall = v.shortfall + push_eps
            positions[i] -= direction * (shortfall / 2.0)
            positions[j] += direction * (shortfall / 2.0)

        # 推出边界的机组投影回边界，并加微小扰动避免卡在角点。
        for k in range(n):
            if not boundary.contains_point(positions[k]):
                positions[k] = boundary.project_to_boundary(positions[k])
                perturbation = rng.uniform(-5.0, 5.0, 2)
                positions[k] += perturbation
                if not boundary.contains_point(positions[k]):
                    positions[k] = boundary.project_to_boundary(positions[k])

    valid, violations = constraint.check(positions)
    inside = boundary.contains_all(positions)
    if not (valid and bool(inside.all())):
        raise RuntimeError(
            f"无法在 {max_iterations} 次迭代内使布局满足间距与边界约束"
            f"（剩余违规 {len(violations)} 对，场外机组 "
            f"{int(np.sum(~inside))} 台），场地可能过于拥挤"
        )

    return positions


def enforce_min_spacing(
    positions: np.ndarray,
    min_distance: float,
    boundary,
    rng: Optional[np.random.Generator] = None,
    max_iterations: int = 1000,
) -> np.ndarray:
    """径向最小距离修复（向后兼容包装）。

    内部通过 :class:`SpacingConstraint` 与 :func:`enforce_spacing`
    实现，因而具有相同的容差/零距离定义与有界退出保证。
    """
    constraint = SpacingConstraint.radial(
        rotor_diameters=np.ones(positions.shape[0]),
        min_distance=min_distance,
    )
    return enforce_spacing(
        positions,
        constraint,
        boundary,
        rng=rng,
        max_iterations=max_iterations,
    )
