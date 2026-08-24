"""The per-point budget on a scan.

A scan writes ONE result file, at the end, so a worker killed on its last and
largest point loses every point before it. On the n_zernike board that is
precisely the expected outcome -- the numerical gradient's cost grows as P^2,
so the axis is meant to run out of affordability somewhere along it -- and the
running-out is the measurement, not something that may also destroy the evidence
for it. These tests pin the containment rather than the timing.
"""
import time

import pytest

from dragrace import worker


def test_deadline_interrupts_and_names_the_budget():
    with pytest.raises(worker.PointTimeout, match="0.2"):
        with worker._deadline(0.2):
            deadline = time.perf_counter() + 10.0
            while time.perf_counter() < deadline:
                pass


def test_deadline_is_disarmed_on_the_way_out():
    """A leaked itimer would fire during a LATER point and be recorded against
    the wrong scan value -- a wrong number rather than a missing one."""
    with worker._deadline(0.2):
        pass
    time.sleep(0.4)          # would have fired by now if it were still armed


def test_deadline_does_not_disturb_a_point_that_finishes():
    with worker._deadline(30.0):
        result = sum(range(1000))
    assert result == 499500


def test_zero_budget_is_no_budget():
    """execution.timeout_s = 0 must mean 'unbounded', not 'fire immediately'."""
    with worker._deadline(0.0):
        time.sleep(0.05)
