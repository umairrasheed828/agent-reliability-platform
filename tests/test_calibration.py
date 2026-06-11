from src.scoring.calibration import compare_judges


def test_aligned_when_judges_agree() -> None:
    v = compare_judges([5, 4, 5, 5], [5, 4, 5, 4], [5, 5, 5, 5], [5, 5, 5, 5])
    assert v.faithfulness_mae <= 0.6
    assert v.aligned


def test_diverged_when_live_is_lenient() -> None:
    # live says 5s, anchor says the answers are actually weak -> big gap
    v = compare_judges([5, 5, 5, 5], [2, 3, 2, 3], [5, 5, 5, 5], [5, 5, 5, 5])
    assert v.faithfulness_mae > 0.6
    assert not v.aligned
