"""Audit-only proposal. Not imported by registered tasks or training entrypoints.

Horizontal squared cost has no saturation. Vertical cost and unfinished cost
retain the prior proposal. Both the raw result and prior zero-state-score floor
are exposed: the latter can hide the intended quadratic gradient.
"""
from dataclasses import dataclass


def errors(xy, tcp_height):
    return max(float(xy) - .017, 0.), max(.05 - float(tcp_height), float(tcp_height) - .06, 0.)


def components(xy_error, height_error):
    return dict(unfinished=.6, horizontal=.8 * (xy_error / .05) ** 2,
                vertical=.8 * height_error / (.04 + height_error))


@dataclass
class Ledger:
    active: bool = False
    qualified_once: bool = False
    locked: bool = False
    detach_xy: float = 0.
    detach_height: float = 0.
    locked_xy: float = 0.
    locked_height: float = 0.

    def observe(self, *, carried, first_loss, release_event, release_paid, xy, tcp_height):
        self.active |= bool(carried)
        ex, ez = errors(xy, tcp_height)
        if first_loss:
            self.detach_xy, self.detach_height = ex, ez
        # Only an accepted first release ends this proposal's cost. An invalid
        # release stores the first-loss error, not the cube's later floor pose.
        if release_event and not release_paid and not self.locked:
            self.locked = True
            self.locked_xy, self.locked_height = self.detach_xy, self.detach_height
        self.qualified_once |= bool(release_paid)

    def cost(self, xy, tcp_height):
        if not self.active or self.qualified_once:
            return 0.
        ex, ez = ((self.locked_xy, self.locked_height) if self.locked else errors(xy, tcp_height))
        return sum(components(ex, ez).values())


def apply_cost(cost, state_score, *, floor_state_score):
    return min(cost, max(float(state_score), 0.)) if floor_state_score else cost


def discounted(values, gamma=.99):
    return sum(gamma ** i * float(v) for i, v in enumerate(values))
