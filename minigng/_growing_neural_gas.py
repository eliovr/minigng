from typing import Any, Optional
import numpy as np
import numpy.typing as npt
import random


class Unit:
    """Represents a neuron/unit in the Growing Neural Gas network.
    
    Attributes:
        prototype: The position vector of this unit in input space.
        error: Accumulated error for this unit.
        neighbors: Set of neighboring units connected by edges.
        id: Unique identifier assigned after training.
        count: Number of data points closest to this unit.
        class_proba: Probability distribution over classes (for classification).
    """
    
    __slots__ = ('prototype', 'error', 'neighbors', 'id', 'count', 'class_proba')
    
    def __init__(self, prototype: npt.NDArray[np.float64], error: float = 0.0) -> None:
        self.prototype = prototype
        self.error = error
        self.neighbors: set['Unit'] = set()

        self.id: int = -1
        self.count: int = 0
        self.class_proba: Optional[dict[str, float]] = None

    def move_towards(self, vector: npt.NDArray[np.float64], eps: float) -> None:
        """Move this unit's prototype towards a target vector.
        
        Args:
            vector: Target vector to move towards.
            eps: Learning rate (fraction of distance to move).
        """
        self.prototype += (vector - self.prototype) * eps

    def predict_proba(self) -> Optional[tuple[str | int, float]]:
        """Return the predicted class with its probability.

        Returns:
            Tuple of (class, probability) or None if not used for classification.
        """
        if self.class_proba:
            return max(self.class_proba.items(), key=lambda e: e[1])
        return None
    
    def predict(self) -> Optional[str | int]:
        """Returns the predicted class for this unit.

        Returns:
            Predicted class or None if not used for classification.
        """
        proba = self.predict_proba()
        return proba[0] if proba else None


class Edge:
    """Represents an edge connecting two units in the network.
    
    Attributes:
        source: One endpoint of the edge.
        target: Other endpoint of the edge.
        age: Number of signals since this edge was last refreshed.
    """
    
    __slots__ = ('source', 'target', 'age')
    
    def __init__(self, source: Unit, target: Unit, age: int = 0) -> None:
        self.source = source
        self.target = target
        self.age = age

    def connects_unit(self, unit: Unit) -> bool:
        """Check if this edge connects to the given unit.
        
        Args:
            unit: Unit to check.
            
        Returns:
            True if the edge connects to the unit.
        """
        return unit in (self.source, self.target)

    def connects_units(self, a: Unit, b: Unit) -> bool:
        """Check if this edge connects the two given units.
        
        Args:
            a: First unit.
            b: Second unit.
            
        Returns:
            True if the edge connects both units.
        """
        return self.connects_unit(a) and self.connects_unit(b)

    def get_partner(self, unit: Unit) -> Unit:
        """Get the other unit connected by this edge.
        
        Args:
            unit: One endpoint of the edge.
            
        Returns:
            The other endpoint.
        """
        return self.target if unit == self.source else self.source


class MiniGNG:
    def __init__(
            self,
            n_epochs: int = 50,
            sigma: int = 100,
            max_units: int = 100,
            eps_b: float = .2,
            eps_n: float = .006,
            max_edge_age: int = 50,
            alpha: float = .5,
            d: float = .995,
            untangle: bool = False,
            max_size_connect: int = 3,
            shuffle: bool = True,
            sample: float = 1.0
        ) -> None:
        """
        Parameters
        ----------
        n_epochs : int (default=50)
            Number of epochs (i.e., runs over the entire data) before stopping.

        sigma : float (default=100)
            How many signals before a new units is added.

        max_units : int (default=100)
            Maximum number of units that will be added.

        eps_b : float (default=.2)
            Adaptation step size for the winning unit.

        eps_n : float (default=.006)
            Adaptation step size for the neighbors of the winning unit.

        max_edge_age : int (default=50)
            Maximum age of the edges, i.e., number of signals before an edge
            is remove. Age is reset for edges connecting the first and
            second winning units.

        alpha : float (default=.5)
            Error reduction rate for the neighbors of a newly created unit.

        d : float (default=.995)
            Error reduction rate for all nodes.

        untangle : boolean (default=False)
            Whether to apply the untangling mechanism, i.e., avoids creating too
            many edges. This may help create better cluster separations.

        max_size_connect : int (default=3)
            States the size of the network the first or second winning units need
            to belong to, in order to allow them to connect (see step 6).
            Used only when untangle=True. Set to 0 to skip this check.

        shuffle : boolean (default=True)
            Whether to shuffles the training data every epoch.

        sample : float (default=1.0)
            Sample a fraction of the training data for every epoch. Takes values
            between 0 and 1 (where 1 = the entire dataset).
        """

        self.n_epochs = n_epochs
        self.sigma = sigma
        self.max_units = max_units
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.max_edge_age = max_edge_age
        self.alpha = alpha
        self.d = d
        self.untangle = untangle
        self.max_size_connect = max_size_connect
        self.shuffle = shuffle
        self.sample = sample
        
        self.units: list[Unit] = []
        self.edges: list[Edge] = []
        self._edge_map: dict[tuple[Unit, Unit], Edge] = {}  # Fast edge lookup
        self.signal_counter: int = 0
        self.classes = None


    def get_params(self, deep: bool = True) -> dict[str, int | float | bool]:
        """
        Get parameters (adapted from scikit-learn's BaseEstimator class).
        Useful for running tests using scikit-learn.
        """
        param_names = ['n_epochs', 'sigma', 'max_units', 'eps_b', 'eps_n', 'max_edge_age',
            'alpha', 'd', 'untangle', 'max_size_connect', 'shuffle', 'sample']

        out = dict()
        for key in param_names:
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out


    def set_params(self, **params: int | float | bool):
        """
        Set the parameters (taken from scikit-learn's BaseEstimator class).
        Useful for running tests using scikit-learn.
        """
        from collections import defaultdict

        if not params:
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

    def _add_edge(self, source: Unit, target: Unit, age: int = 0) -> Edge:
        """Add an edge to the network with map tracking.
        
        Args:
            source: First endpoint.
            target: Second endpoint.
            age: Initial age of the edge.
            
        Returns:
            The created edge.
        """
        edge = Edge(source, target, age)
        self.edges.append(edge)
        # Store in both directions for fast lookup
        self._edge_map[(source, target)] = edge
        self._edge_map[(target, source)] = edge
        return edge
    
    def _remove_edge(self, edge: Edge) -> None:
        """Remove an edge from the network and update map.
        
        Args:
            edge: The edge to remove.
        """
        if edge in self.edges:
            self.edges.remove(edge)
        # Remove from map in both directions
        self._edge_map.pop((edge.source, edge.target), None)
        self._edge_map.pop((edge.target, edge.source), None)
    
    def _get_edge(self, a: Unit, b: Unit) -> Optional[Edge]:
        """Get edge between two units in O(1) time.
        
        Args:
            a: First unit.
            b: Second unit.
            
        Returns:
            Edge if exists, None otherwise.
        """
        return self._edge_map.get((a, b))


    def init_model(self, X: npt.NDArray[np.float64]) -> None:
        """Initialize the model with two random units from the data.
        
        Args:
            X: Training data array of shape (n_samples, n_features).
            
        Raises:
            AssertionError: If X is not a 2D array.
        """
        assert X.ndim == 2, f'Expected array of 2 dimensions, got {X.ndim}'
        n = len(X)
        
        # Ensure we pick two different samples for initialization
        if n >= 2:
            idx_a, idx_b = random.sample(range(n), 2)
        else:
            # Edge case: only one sample - add small random perturbation
            idx_a = idx_b = 0
            
        a = Unit(X[idx_a].copy())
        b_proto = X[idx_b].copy()
        
        # If same sample was selected, add small perturbation to avoid identical units
        if idx_a == idx_b:
            b_proto = b_proto + np.random.normal(0, 0.01, b_proto.shape)
            
        b = Unit(b_proto)

        a.neighbors.add(b)
        b.neighbors.add(a)
        self.units = [a, b]
        self._add_edge(a, b)


    def predict(self, X: npt.NDArray[np.float64]) -> tuple[list[int], Optional[list[str | int]]]:
        """Returns the unit id to which each data point is closest to and, if
        used for classification, the class predicted by each unit.

        Args:
            X: Data to predict of shape (n_samples, n_features).

        Returns:
            Tuple of (unit_ids, classes). Classes is None when not used for
            classification (calling `fit` without `y`).
            
        Raises:
            AssertionError: If X is not a 2D array.
        """
        assert X.ndim == 2, f'Expected array of 2 dimensions, got {X.ndim}'
        if len(self.units) == 0:
            return [], None

        # Cache prototypes array to avoid repeated creation
        active_units = [u for u in self.units if u.count > 0]
        if not active_units:
            return [], None
            
        prototypes = np.array([u.prototype for u in active_units])
        unit_ids = []
        labels = []

        for x in X:
            dists = np.linalg.norm(x - prototypes, axis=1)
            unit_id = np.argmin(dists)
            unit_ids.append(unit_id)
            unit = active_units[unit_id]
            if unit.class_proba:
                labels.append(unit.predict())

        return unit_ids, (labels if labels else None)


    def fit(self, X: npt.NDArray[np.float64], y: Optional[npt.NDArray] = None) -> 'MiniGNG':
        """Train the Growing Neural Gas model.
        
        Args:
            X: Training data of shape (n_samples, n_features).
            y: Optional labels for classification of shape (n_samples,).
            
        Returns:
            Self for method chaining.
        """
        # Train GNG
        for _ in range(self.n_epochs):
            self.partial_fit(X)
        
        # Assign IDs and compute prototypes once
        for i, u in enumerate(self.units):
            u.id = i
            
        prototypes = np.array([u.prototype for u in self.units])
        
        # Compute distances efficiently based on dataset size
        # For small datasets, use vectorized computation; for large ones, use row-by-row
        # Note: scipy.spatial.distance.cdist would be more memory-efficient but is not a dependency
        n_samples, n_features = X.shape
        n_units = len(self.units)
        # Estimate memory usage: n_samples * n_units * n_features * 8 bytes
        estimated_memory_mb = (n_samples * n_units * n_features * 8) / (1024 * 1024)
        
        if estimated_memory_mb < 100:  # Less than 100MB
            # Use vectorized computation for better performance
            distances = np.linalg.norm(X[:, np.newaxis, :] - prototypes[np.newaxis, :, :], axis=2)
            unit_assignments = np.argmin(distances, axis=1)
        else:
            # Use row-by-row for memory efficiency on large datasets
            unit_assignments = np.zeros(len(X), dtype=int)
            for i, x in enumerate(X):
                dists = np.linalg.norm(prototypes - x, axis=1)
                unit_assignments[i] = np.argmin(dists)
        
        # Group samples by unit
        groups = {i: [] for i in range(len(self.units))}
        for sample_idx, unit_id in enumerate(unit_assignments):
            groups[unit_id].append(sample_idx)

        if y is not None:
            self.classes = np.unique(y)

        for unit_id, sample_indices in groups.items():
            unit = self.units[unit_id]
            unit.count = len(sample_indices)
            
            if y is not None and sample_indices:
                unit.class_proba = {c: 0.0 for c in self.classes}
                unique, counts = np.unique(y[sample_indices], return_counts=True)
                unit.class_proba.update(dict(zip(unique, counts / unit.count)))

        return self
        

    def fit_predict(
            self,
            X: npt.NDArray[np.float64],
            return_unit_ids: bool = False) -> list[Any] | tuple[list[Any], list[int]]:
        """Fit the model and return predictions.
        
        Args:
            X: Training data.
            return_unit_ids: Whether to return unit IDs along with predictions.
            
        Returns:
            For clustering (no y): returns unit_ids or (unit_ids, unit_ids)
            For classification (with y): returns predictions or (predictions, unit_ids)
        """
        self.fit(X)
        unit_ids, predictions = self.predict(X)
        
        if return_unit_ids:
            # Return (predictions/unit_ids, unit_ids)
            return (predictions if predictions is not None else unit_ids), unit_ids
        # Return predictions if available, otherwise unit_ids
        return predictions if predictions is not None else unit_ids


    def partial_fit(self, X: npt.NDArray[np.float64]) -> None:
        """Perform one epoch of training on the data.
        
        Args:
            X: Training data of shape (n_samples, n_features).
        """
        if len(self.units) == 0:
            self.init_model(X)

        # Cache parameters as local variables for faster access
        sigma = self.sigma
        alpha = self.alpha
        eps_b = self.eps_b
        eps_n = self.eps_n
        d = self.d
        max_edge_age = self.max_edge_age
        max_units = self.max_units

        signals = X

        if self.shuffle or self.sample < 1.0:
            size = len(X)
            n_samples = int(size * self.sample) if self.sample < 1.0 else size
            # Use permutation for efficient shuffling/sampling
            indices = np.random.permutation(size)[:n_samples]
            signals = X[indices]

        # Process each signal (note: prototypes must be recreated each iteration
        # as they are updated via move_towards())
        for signal in signals:
            self.signal_counter += 1

            # 2. Find the nearest unit S1 and the second-nearest unit S2.
            # Using argpartition for O(n) instead of argsort's O(n log n)
            prototypes = np.array([u.prototype for u in self.units])
            distances = np.linalg.norm(prototypes - signal, axis=1)
            
            # Get indices of two smallest distances efficiently
            if len(self.units) >= 2:
                # argpartition(arr, 1) partitions so that arr[0] is smallest,
                # and arr[1] contains the second smallest (but not sorted)
                # We take indices [0:2] and sort them to ensure correct ordering
                indices = np.argpartition(distances, 1)[:2]
                # Sort these two indices by their distances to get nearest first
                sorted_indices = indices[np.argsort(distances[indices])]
                unit_a_id, unit_b_id = sorted_indices[0], sorted_indices[1]
            else:
                # Edge case: only one unit (shouldn't happen in normal operation)
                unit_a_id = unit_b_id = 0
                
            unit_a: Unit = self.units[unit_a_id]
            unit_b: Unit = self.units[unit_b_id] if len(self.units) >= 2 else unit_a
            dist = distances[unit_a_id]

            # 3. Increment the age of all edges emanating from S1 and find edge to S2.
            # Use edge map for O(1) lookup instead of O(n) search
            ab_edge = self._get_edge(unit_a, unit_b)
            
            for neighbor in unit_a.neighbors:
                edge = self._get_edge(unit_a, neighbor)
                if edge:
                    edge.age += 1

            # 4. Add the squared distance between the input signal and
            # the nearest unit in input space to a local counter variable.
            unit_a.error += dist * dist

            # 5. Move S1 and its direct topological neighbors towards E by
            # fractions Eb and En, respectively, of the total distance.
            unit_a.move_towards(signal, eps_b)
            for neighbor in unit_a.neighbors:
                neighbor.move_towards(signal, eps_n)

            # 6. If S1 and S2 are connected by an edge, set the age of this
            # edge to zero. If such an edge does not exist, create it.
            if ab_edge is not None:
                ab_edge.age = 0
            elif not self.untangle or self.no_curling(unit_a, unit_b):
                unit_a.neighbors.add(unit_b)
                unit_b.neighbors.add(unit_a)
                self._add_edge(unit_a, unit_b)

            # 7. Remove edges with an age larger than maxAge. If this results in
            # points having no emanating edges, remove them as well.
            edges_to_remove = []
            units_to_remove = set()

            for e in self.edges:
                if e.age > max_edge_age:
                    edges_to_remove.append(e)
                    
            for e in edges_to_remove:
                # Edge is too old, remove it
                e.source.neighbors.discard(e.target)
                e.target.neighbors.discard(e.source)
                self._remove_edge(e)

                # Mark units with no neighbors for removal
                if len(e.source.neighbors) == 0:
                    units_to_remove.add(e.source)
                if len(e.target.neighbors) == 0:
                    units_to_remove.add(e.target)
            
            # Remove isolated units after all edge processing is complete
            for unit in units_to_remove:
                self.units.remove(unit)

            # 8. If the number of input signals generated so far is an integer
            # multiple of a parameter A, insert a new unit as follows.
            if self.signal_counter % sigma == 0 and len(self.units) < max_units:

                # Determine the unit q with the maximum accumulated error.
                q = max(self.units, key=lambda u: u.error)

                # Insert a new unit r halfway between q and its neighbor f with
                # the largest error variable.
                f = max(q.neighbors, key=lambda u: u.error)

                new_prototype = (q.prototype + f.prototype) * 0.5
                r = Unit(new_prototype)
                self.units.append(r)

                # Insert edges connecting the new unit r with units q and f,
                # and remove the original edge between q and f.
                # Remove old edge first to maintain consistency
                qf_edge = self._get_edge(q, f)
                if qf_edge:
                    self._remove_edge(qf_edge)
                
                q.neighbors.discard(f)
                f.neighbors.discard(q)

                q.neighbors.add(r)
                f.neighbors.add(r)
                r.neighbors.add(q)
                r.neighbors.add(f)
                    
                self._add_edge(q, r)
                self._add_edge(f, r)

                q.error *= alpha
                f.error *= alpha
                r.error = q.error

            # 9. Decrease all error variables by multiplying them with a constant d.
            for u in self.units:
                u.error *= d


    def score(self, X: npt.NDArray[np.float64], y: npt.NDArray) -> float:
        """Calculate accuracy score for classification.
        
        Args:
            X: Test data.
            y: True labels.
            
        Returns:
            Accuracy score (1.0 - error rate).
        """
        _, predictions = self.predict(X)
        if predictions is None:
            raise ValueError("Model not trained for classification")
        diff = np.array(predictions) != y
        score = np.sum(diff) / len(y)
        return 1.0 - score


    def network_size_compare(self, node: Unit, size: int) -> int:
        """
        Check the size of the network a node belongs to against a threshold.
        
        Args:
            node: The unit to start the search from.
            size: The size threshold to compare against.
            
        Returns:
            1 if network size > size, -1 if network size < size, 0 if equal.
        """
        open_nodes = node.neighbors.copy()
        closed_nodes = {node}
        n_nodes = 1

        while open_nodes and n_nodes <= size:
            closed_nodes.update(open_nodes)
            n_nodes = len(closed_nodes)

            if n_nodes <= size:
                aux = set()
                for n in open_nodes:
                    aux.update(ne for ne in n.neighbors if ne not in closed_nodes)
                open_nodes = aux
        
        if n_nodes < size:
            return -1
        elif n_nodes > size:
            return 1
        else:
            return 0

    def exists_path(self, a: Unit, b: Unit) -> bool:
        """
        Check if there's a path between two units in the network.
        
        Args:
            a: Starting unit.
            b: Target unit.
            
        Returns:
            True if a path exists, False otherwise.
        """
        if a == b:
            return True
            
        open_nodes = a.neighbors.copy()
        closed_nodes = {a}

        while open_nodes:
            if b in open_nodes:
                return True

            closed_nodes.update(open_nodes)
            aux = set()
            for n in open_nodes:
                aux.update(ne for ne in n.neighbors if ne not in closed_nodes)
            open_nodes = aux
            
        return False

    def no_curling(self, a: Unit, b: Unit) -> bool:
        """
        Check if connecting units a and b would 'curl' the network.
        
        Prevents creating high-dimensional graph structures by checking
        the topology before adding an edge.
        
        Args:
            a: First unit.
            b: Second unit.
            
        Returns:
            True if connection is allowed (no curling), False otherwise.
        """
        bridges = a.neighbors & b.neighbors
        n_bridges = len(bridges)

        if n_bridges == 2:
            # No curling if the two common neighbors are not connected.
            bridge_list = list(bridges)
            x, y = bridge_list[0], bridge_list[1]
            return len(x.neighbors & y.neighbors) == 0

        elif n_bridges == 1:
            # No curling if there are less than 2 common neighbors between
            # 'a' and 'x', and between 'b' and 'x'.
            [x] = bridges
            xn = x.neighbors
            an = a.neighbors
            bn = b.neighbors

            return len(an & xn) < 2 and len(bn & xn) < 2 and len(xn) <= 6

        elif n_bridges == 0:
            has_min_size = (
                self.max_size_connect <= 0 or
                self.network_size_compare(b, self.max_size_connect) < 1
            )

            return has_min_size and not self.exists_path(a, b)

        return False


    def save_vna(self, filename: str) -> None:
        """
        Save graph (GNG model) to .vna format.
        
        The .vna format can be loaded into visualization tools like Gephi.
        
        Args:
            filename: Path to the output file.
        """
        nodes = '*node data\nID name\n'
        nodes += '\n'.join([f'{i} {u.predict() or i}' for i, u in enumerate(self.units)])

        edges = '*tie data\nfrom to strength\n'
        edges += '\n'.join([
            f'{self.units.index(e.source)} {self.units.index(e.target)} 1'
            for e in self.edges])

        graph = f'{nodes}\n{edges}'

        with open(filename, 'w') as out:
            out.write(graph.strip())


    def save_gml(self, filename: str) -> None:
        """
        Save graph (GNG model) to .gml format.
        
        The .gml format can be loaded into visualization tools like Gephi.
        
        Args:
            filename: Path to the output file.
        """
        nodes = [
        """
        node
        [
          id {i}
          label {label}
        ]
        """.format(i=i, label=u.predict() or i) for i, u in enumerate(self.units)]

        edges = [
        """
        edge
        [
          source {s}
          target {t}
        ]
        """.format(s=self.units.index(e.source), t=self.units.index(e.target)) for e in self.edges]

        graph = """
        graph
        [
          {ns}
          {es}
        ]
        """.format(ns='\n'.join(nodes), es='\n'.join(edges))

        with open(filename, 'w') as out:
            out.write(graph)
