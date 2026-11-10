"""Retain the first terminal outcome, distinguishing success from failure."""
import jax
import jax.numpy as jp


def latch(previous,updated,terminal,failed):
    # No post-terminal state or metric may overwrite a recorded outcome.
    def select(old,new):
        # Warp also carries shared/empty contact buffers without a world axis.
        # Keep those fresh; freeze only per-world state and outcome arrays.
        if new.ndim < terminal.ndim or new.shape[:terminal.ndim] != terminal.shape:
            return new
        return jp.where(terminal.reshape(terminal.shape+(1,)*(new.ndim-terminal.ndim)),old,new)
    updated=jax.tree.map(select,previous,updated)
    active=~terminal
    ended=updated.done > 0
    failed=failed | (active & ended & (updated.metrics['success'] <= 0))
    return updated,terminal | ended,failed,active
