#pragma once

namespace drake {
namespace multibody {
/** For a plane aᵀx = b, we denote the side {x | aᵀx ≥ b} as "positive" side of
 * the plane, as it is on the same direction of the plane normal vector a. The
 * side { x |aᵀx ≤ b} as the "negative" side of the plane.
 */
enum class PlaneSide {
  kPositive,
  kNegative,
};
}  // namespace multibody
}  // namespace drake
