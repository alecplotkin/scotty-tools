import pytest
import numpy as np
import numpy.testing as npt
from scotty.simulate.latents import (
    Clone,
    CloneForest,
    CloneTrajectory,
    Simulation,
    standard_normal_init,
)


class TestClone:
    def test_initialization(self, clone):
        assert clone.is_alive()

    def test_divide(self, clone):
        clone.divide(children=[1, 2], time=0.5)
        assert len(clone.children) == 2
        assert clone.death_time == 0.5
        assert not clone.is_alive()

    def test_is_alive_with_time(self, clone):
        clone.divide(children=[1, 2], time=0.5)
        assert not clone.is_alive(0.55)
        assert clone.is_alive(0.45)


class TestCloneForest:
    def test_initialization(self, clone_forest):
        assert len(clone_forest) == 5

    def test_division(self, clone_forest):
        clone_forest.division(0, time=0.5)
        assert len(clone_forest) == 7
        assert not clone_forest[0].is_alive()
        assert clone_forest[5].parent == 0

    def test_death(self, clone_forest):
        clone_forest.death(1, time=0.6)
        assert not clone_forest[1].is_alive()
        assert clone_forest[1].is_alive(time=0.5)

    def test_get_descendants(self, clone_forest):
        clone_forest.division(0, time=0.5)
        clone_forest.death(6, time=0.6)
        assert clone_forest.get_descendants(0, 0.55) == [5, 6]
        assert clone_forest.get_descendants(0, 0.7) == [5]


class TestCloneTrajectory:
    def test_initialization(self, clone_trajectory):
        X = clone_trajectory.X[-1]
        assert X.shape == (5, 2)
        ix = clone_trajectory.indices[-1]
        assert ix.shape == (5, )

    def test_step(self, clone_trajectory):
        X = clone_trajectory.X[-1]
        ix = clone_trajectory.indices[-1]
        clone_trajectory.step(X, ix, time=0.5)
        assert len(clone_trajectory.X) == 1

    def test_step_with_update(self, clone_trajectory):
        X = clone_trajectory.X[-1]
        ix = clone_trajectory.indices[-1]
        clone_trajectory.step(X, ix, time=0.5, update=True)
        assert len(clone_trajectory.X) == 2
        assert len(clone_trajectory.lineages) == 5

    def test_step_with_birth_death(self, clone_trajectory):
        """Verify CloneTrajectory.step correctly handles birth and death updates."""

        X = clone_trajectory.X[-1]
        ix = clone_trajectory.indices[-1]

        birth_mask = np.zeros_like(ix, dtype=bool)
        birth_mask[0:2] = True  # Two clones should divide.
        death_mask = np.zeros_like(ix, dtype=bool)
        death_mask[-1] = True  # One clone should die.

        clone_trajectory.step(X, ix, time=0.5, update=True, birth_mask=birth_mask, death_mask=death_mask)
        assert len(clone_trajectory.X) == 2
        assert len(clone_trajectory.lineages) == 9  # This should reflect the 4 new clones.

        X_new = clone_trajectory.X[-1]
        ix_new = clone_trajectory.indices[-1]
        assert X_new.shape[0] == 6  # 5 initial + (4 - 2) births - 1 deaths
        assert len(ix_new) == 6  # 5 initial + (4 - 2) births - 1 deaths

        test_parent_ix = clone_trajectory.lineages[ix_new[-1]].parent
        npt.assert_array_equal(X_new[-1, :], X[test_parent_ix, :])
        assert not clone_trajectory.lineages[test_parent_ix].is_alive()
        assert clone_trajectory.lineages[test_parent_ix].is_alive(0.5)


@pytest.mark.skip(reason="Test for Simulation not implemented yet")
class TestSimulation:
    pass
