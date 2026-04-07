import numpy as np
import numpy.typing as npt

from dataclasses import dataclass
from typing import List, Callable
from tqdm import tqdm


@dataclass
class Clone:
    parent: int
    birth_time: float
    death_time: float = None
    children: List[int] = None

    def __post_init__(self):
        self.children = []

    def divide(self, children: List[int], time: float):
        self.children = children
        self.death_time = time

    def die(self, time: float):
        self.death_time = time

    def is_alive(self, time: float = None):
        has_death = self.death_time is not None
        if time is None:
            return not has_death
        elif has_death and (time >= self.death_time):
            return False
        else:
            return True


@dataclass
class CloneForest:
    init_size: int = 10

    def __post_init__(self):
        clones = []
        for _ in range(self.init_size):
            clones.append(Clone(parent=None, birth_time=0))
        self.clones = clones

    def __getitem__(self, item):
        return self.clones[item]

    def __len__(self):
        return len(self.clones)

    def division(self, ix: int, time: float):
        clone = self[ix]
        if not clone.is_alive():
            raise ValueError("Clone is dead.")
        max_ix = len(self.clones)
        children = [max_ix, max_ix + 1]
        clone.divide(children, time)
        for child_ix in children:
            child = Clone(parent=ix, birth_time=time)
            self.clones.append(child)

    def death(self, ix: int, time: float):
        self[ix].die(time)

    def get_descendants(self, ix: int, time: float):
        clone = self[ix]
        if clone.is_alive(time):
            return [ix]
        descendants = []
        for ix_child in clone.children:
            descendants.extend(self.get_descendants(ix_child, time))
        return descendants


@dataclass
class CloneTrajectory:
    init_size: int = 10
    ndim: int = 2
    init_fun: Callable = None

    def __post_init__(self):
        if self.init_fun is None:
            self.init_fun = np.zeros
        X_init = self.init_fun((self.init_size, ))
        self.X = [X_init]
        self.indices = [np.arange(self.init_size)]
        self.lineages = CloneForest(self.init_size)
        self.times = [0]

    def update(self, X: npt.NDArray, indices: npt.NDArray, time: float):
        self.X.append(X)
        self.indices.append(indices)
        self.times.append(time)

    def step(
        self,
        X: npt.NDArray,
        indices: npt.NDArray,
        time: float,
        birth_mask: npt.NDArray = None,
        death_mask: npt.NDArray = None,
        update: bool = False,
    ):
        if birth_mask is None:
            birth_mask = np.zeros_like(indices, dtype=bool)
        if death_mask is None:
            death_mask = np.zeros_like(indices, dtype=bool)
        mask = ~(birth_mask | death_mask)

        # Handle births/deaths for X
        X_birth = X[birth_mask, :]
        X = X[mask, :]
        X_step = np.concatenate((X, X_birth, X_birth))

        # Handle births/deaths for ix
        ix_birth = indices[birth_mask]
        ix_death = indices[death_mask]
        indices = indices[mask]
        ix_children = len(self.lineages) + 2 * np.arange(len(ix_birth))
        ix_step = np.concatenate((indices, ix_children, ix_children + 1))

        # Update lineages with births/deaths
        for ix in ix_death:
            self.lineages.death(ix, time)
        for ix in ix_birth:
            self.lineages.division(ix, time)

        if update:
            self.update(X_step, ix_step, time)
        return X_step, ix_step, time

    def get_coupling(self, t0: float, t1: float):
        if t0 in self.times:
            ix0 = self.indices[np.argwhere(np.array(self.times) == t0).squeeze()]
        else:
            raise NotImplementedError("Method for finding initial indices not implemented for arbitrary times.")

        coupling_dict = {ix: self.lineages.get_descendants(ix, t1) for ix in ix0}
        return coupling_dict

    def get_tags(self):
        times = self.times
        n_times = len(times)
        clone_tags = dict()
        for i in range(n_times - 1):
            tags_t = dict()
            for j in range(i, n_times):
                coupling = self.get_coupling(times[i], times[j])
                rev_coupling = dict()
                for k, v in coupling.items():
                    for ix in v:
                        rev_coupling[ix] = k
                tags_t.update(rev_coupling)
            clone_tags[times[i]] = tags_t
        return clone_tags


@dataclass
class Simulation:
    diffusivity: float
    ndim: int = 2
    drift: Callable = None
    birth: Callable = None
    death: Callable = None
    init_fun: Callable = None

    def __post_init__(self):
        if self.drift is None:
            self.drift = np.zeros_like
        if self.birth is None:
            self.birth = lambda X: np.zeros(X.shape[0])
        if self.death is None:
            self.death = lambda X: np.zeros(X.shape[0])

    def simulate(
        self,
        dt: float,
        n_steps: int,
        update_every: int = 10,
        init_size: int = 100,
        seed: int = None,
        dt_precision: int = 4,
    ):
        traj = CloneTrajectory(
            init_size=init_size,
            ndim=self.ndim,
            init_fun=self.init_fun,
        )
        X = traj.X[0]
        ix = traj.indices[0]
        time = 0.0

        self._rng = np.random.default_rng(seed)
        for step in tqdm(range(n_steps)):
            update = (step + 1) % update_every == 0
            time = round(time + dt, dt_precision)  # So that we don't accumulate floating point errors.
            # Update positions, births, deaths, using vectorized operations.
            X_step, birth_mask, death_mask = self.step(X, ix, dt)
            # Update lineages and resize X based on births and deaths.
            X, ix, _ = traj.step(X_step, ix, time, birth_mask, death_mask, update)
        return traj

    def step(
        self,
        X: npt.NDArray,
        ix: npt.NDArray,
        dt: float,
    ):
        mu = self.drift(X)
        dW = self._rng.normal(0, 1, X.shape)
        X_step = X - mu * dt + np.sqrt(self.diffusivity) * dW
        p_birth = self.birth(X_step) * dt
        p_death = self.death(X_step) * dt
        rv = self._rng.uniform(size=len(ix))
        birth_mask = (rv < p_birth)
        death_mask = (rv > 1 - p_death)
        if any(birth_mask & death_mask):
            raise ValueError("Conflicting birth and death events. Try decreasing dt.")
        return X_step, birth_mask, death_mask
