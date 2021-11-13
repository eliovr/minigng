import numpy as np
import random


class Unit:
    def __init__(self, prototype, error=.0):
        self.prototype = prototype
        self.error = error
        self.neighbors = set()

        self.count = 0
        self.var = None # variance.
        self.cov = None # covariance.
        self.labels = None

    def move_towards(self, vector, eps):
        self.prototype += (vector - self.prototype) * eps

    def get_label(self):
        """
        Returns this unit's label with the highest probability.
        Works only when GNG is used in a supervised manner (by providing 'y' to 'fit').
        """

        label = -1
        max_p = 0

        for k, v in self.labels.items():
            if max_p < v:
                label = k
                max_p = v

        return label


class Edge:
    def __init__(self, source, target, age=0):
        self.source = source
        self.target = target
        self.age = age

    def connects_unit(self, unit):
        return unit in (self.source, self.target)

    def connects_units(self, a, b):
        return self.connects_unit(a) and self.connects_unit(b)

    def get_partner(self, unit):
        if unit == self.source:
            return self.target
        else:
            return self.source


class MiniGNG:
    def __init__(self, n_epochs=50, sigma=100, max_units=100, eps_b=.2,
            eps_n=.006, max_edge_age=50, alpha=.5, d=.995,
            untangle=False, max_size_connect=3, shuffle=True, sample=1.0):
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

        eps_n : float (defailt=.006)
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
        
        self.units = []
        self.edges = []
        self.signal_counter = 0
        self.classes = None


    def get_params(self, deep=True):
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


    def set_params(self, **params):
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


    def init_model(self, X):
        n = len(X) - 1
        a = Unit(X[random.randint(0, n)])
        b = Unit(X[random.randint(0, n)])

        a.neighbors.add(b)
        b.neighbors.add(a)
        self.units = [a, b]
        self.edges.append(Edge(a, b))


    def predict(self, X, return_unit_ids=False):
        if len(self.units) == 0:
            return None

        else:
            groups = {i: [] for i, _ in enumerate(self.units)}
            prototypes = np.array([u.prototype for u in self.units if u.count > 0])
            unit_ids = []
            labels = []     

            for i, x in enumerate(X):
                dists = np.linalg.norm(x - prototypes, axis=1)
                unit_id = np.argmin(dists)
                groups[unit_id].append(i)
                unit_ids.append(unit_id)
                unit = self.units[unit_id]

                if unit.labels is not None:
                    labels.append(unit.get_label())

        if return_unit_ids:
            return labels, unit_ids

        return np.array(labels)


    def fit(self, X, y=None):
        # Train GNG
        for _ in range(0, self.n_epochs):
            self.partial_fit(X)
        
        # Update model with density, variance and labels (if 'y' is given)
        groups = {i: [] for i, _ in enumerate(self.units)}
        prototypes = np.array([u.prototype for u in self.units])
        labels = []

        for i, x in enumerate(X):
            dists = np.linalg.norm(x - prototypes, axis=1)
            unit_id = np.argmin(dists)
            groups[unit_id].append(i)
            labels.append(unit_id)

        for k, v in groups.items():
            unit = self.units[k]
            points_in_unit = X[v]
            unit.count = len(points_in_unit)
            
            if y is not None:
                unique, counts = np.unique(y[v], return_counts=True)
                unit.labels = dict(zip(unique, counts / unit.count))

        self.classes = np.sort(np.unique(y))
        return self
        

    def fit_predict(self, X, return_unit_ids=False):
        self.fit(X)
        return self.predict(X, return_unit_ids)


    def partial_fit(self, X):
        if len(self.units) == 0:
            self.init_model(X)

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
            n_samples = size
            if self.sample < 1.0: n_samples = int(size * self.sample)
            signals = X[np.random.choice(size, n_samples)]

        for signal in signals:
            self.signal_counter += 1

            # 2. Find the nearest unit S1 and the second-nearest unit S2.
            prototypes = np.array([u.prototype for u in self.units])
            distance = np.linalg.norm(prototypes - signal, axis=1)
            units_ids = np.argsort(distance)

            unit_a_id, unit_b_id = units_ids[:2]
            unit_a = self.units[unit_a_id]
            unit_b = self.units[unit_b_id]
            dist = distance[unit_a_id]

            ab_edge = None

            # 3. Increment the age of all edges emanating from S1.
            for e in self.edges:
                if e.connects_unit(unit_a):
                    e.age += 1

                    if e.connects_unit(unit_b):
                        ab_edge = e
                        

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
            if not ab_edge is None:
                ab_edge.age = 0

            elif not self.untangle or self.no_curling(unit_a, unit_b):
                unit_a.neighbors.add(unit_b)
                unit_b.neighbors.add(unit_a)
                self.edges.append(Edge(unit_a, unit_b))

            # 7. Remove edges with an age larger than maxAge. If this results in
            # points having no emanating edges, remove them as well.
            _edges = []

            for e in self.edges:
                if e.age <= max_edge_age:
                    _edges.append(e)

                elif e.age > max_edge_age:
                    e.source.neighbors.remove(e.target)
                    e.target.neighbors.remove(e.source)

                    if len(e.source.neighbors) <= 0:
                        self.units.remove(e.source)

                    if len(e.target.neighbors) <= 0:
                        self.units.remove(e.target)

            self.edges = _edges

            # 8. If the number of input signals generated so far is an integer
            # multiple of a parameter A, insert a new unit as follows.
            if self.signal_counter % sigma == 0 and len(self.units) < max_units:

                # Determine the unit q with the maximum accumulated error.
                q = max(self.units, key=lambda u: u.error)

                # Insert a new unit r halfway between q and its neighbor f with
                # the largest error variable.
                f = max(q.neighbors, key=lambda u: u.error)

                new_prototype = (q.prototype + f.prototype) * .5
                r = Unit(new_prototype)
                self.units.append(r)

                # Insert edges connecting the new unit r with units q and f,
                # and remove the original edge between q and f.
                q.neighbors.remove(f)
                f.neighbors.remove(q)

                q.neighbors.add(r)
                f.neighbors.add(r)
                r.neighbors.add(q)
                r.neighbors.add(f)

                self.edges = [e for e in self.edges if not e.connects_units(q, f)]
                self.edges.append(Edge(q, r))
                self.edges.append(Edge(f, r))

                q.error = q.error * alpha
                f.error = f.error * alpha
                r.error = q.error

            # 9. Decrease all error variables by multiplying them with a constant d.
            for u in self.units:
                u.error *= d


    def score(self, X, y):
        _, predictions = self.predict(X)
        diff = predictions - y
        score = np.count_nonzero(diff) / len(y)
        return 1 - score


    def net_size_compare(self, node, size):
        """
        Checks the size of the network a node belongs to, against the parameter size.
        Returns 1 if the size of the network is greater than "size", -1 if its lower,
        and 0 if equal.
        """

        open_nodes = node.neighbors
        closed_nodes = set()        
        closed_nodes.add(node)
        n_nodes = 1

        while len(open_nodes) > 0 and n_nodes <= size:
            closed_nodes = closed_nodes | open_nodes
            n_nodes = len(closed_nodes)

            if n_nodes <= size:
                aux = set()
                for n in open_nodes:
                    aux = aux | {ne for ne in n.neighbors if not ne in closed_nodes}
                    
                open_nodes = aux
        
        if n_nodes < size: return -1
        elif n_nodes > size: return 1
        else: return 0

    
    def exists_path(self, a, b):
        """
        True if there's a path between a and b, False otherwise.
        """

        open_nodes = a.neighbors
        closed_nodes = set()
        exists = False

        while len(open_nodes) > 0 and not exists:
            exists = b in open_nodes

            if not exists:
                closed_nodes = closed_nodes | open_nodes
                aux = set()
                for n in open_nodes:
                    aux = aux | {ne for ne in n.neighbors if not ne in closed_nodes}
                    
                open_nodes = aux
            
        return exists


    def no_curling(self, a, b):
        """
        Checks if connecting units a and b would 'curl' the network into a
        high-dimensional graph (partially).
        """

        bridges = a.neighbors & b.neighbors
        n_bridges = len(bridges)

        if n_bridges == 2:
            # no curling if the two common neighbors are not connected.
            x, y = bridges
            return len(x.neighbors.intersection(y.neighbors)) == 0

        elif n_bridges == 1:
            # no curling if there are less than 2 common neighbors between
            # 'a' and 'x', and between 'b' and 'x'.
            [x] = bridges
            xn = x.neighbors
            an = a.neighbors
            bn = b.neighbors

            return len(an & xn) < 2 and len(bn & xn) < 2 and len(xn) <= 6

        elif n_bridges == 0:
            has_min_size = (
                self.max_size_connect <= 0 or
                self.net_size_compare(b, self.max_size_connect) < 1
            )

            return has_min_size and not self.exists_path(a, b)

        return False


    def save_vna(self, filename):
        """
        Saves graph (GNG model) to .vna format (which can then be loaded into, e.g., Gephi).
        """

        nodes = '*node data\nID name\n'
        nodes += '\n'.join([f'{i} {u.get_label() if u.labels else i}' for i, u in enumerate(self.units)])

        edges = '*tie data\nfrom to strength\n'
        edges += '\n'.join([
            f'{self.units.index(e.source)} {self.units.index(e.target)} 1'
            for e in self.edges])

        graph = f'{nodes}\n{edges}'

        with open(filename, 'w') as out:
            out.write(graph.strip())


    def save_gml(self, filename):
        """
        Saves graph (GNG model) to .gml format (which can then be loaded into, e.g., Gephi).
        """

        nodes = [
        """
        node
        [
          id {i}
          label {label}
        ]
        """.format(i=i, label=u.get_label() if u.labels else i) for i, u in enumerate(self.units)]

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
