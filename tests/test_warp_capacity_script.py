"""The stress command must not accept silent capacity or numeric failures."""
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "check_warp_capacity", Path(__file__).parents[1] / "scripts/check_warp_capacity.py"
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def test_solver_limit_bits_are_reported_but_not_capacity_failures():
    sample = dict(overflow_bits_or=(1 << 9) | (1 << 10), nonfinite_worlds=0,
                  contacts_total=20, constraints_max_per_world=79, broadphase_pairs_total=30)
    capacities = dict(naconmax=160, njmax=512)
    assert checker.sample_failures(sample, capacities) == []
    sample["overflow_bits_or"] |= 1 << 3
    assert checker.sample_failures(sample, capacities) == ["critical_overflow"]


def test_counts_and_nonfinite_fail_without_an_overflow_flag():
    sample = dict(overflow_bits_or=0, nonfinite_worlds=1, contacts_total=161,
                  constraints_max_per_world=513, broadphase_pairs_total=162)
    assert checker.sample_failures(sample, dict(naconmax=160, njmax=512)) == [
        "nonfinite_state", "contacts_exceed_capacity", "constraints_exceed_capacity",
        "broadphase_exceeds_capacity",
    ]
