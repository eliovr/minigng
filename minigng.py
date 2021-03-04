import numpy as np
from scipy.stats import norm, multivariate_normal
import random


class Unit:
    def __init__(self, prototype, error=.0):
        self.prototype = prototype
        self.error = error
        self.neighbors = set()

        self.count = 0
        self.var = None # variance.
        self.cov = None # covariance.
        self.labels = {}

    def distance_to(self, vector):
        return np.linalg.norm(vector - self.prototype)

    def move_towards(self, vector, eps):
        self.prototype += (vector - self.prototype) * eps


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
            untangle=False, min_net_size=3, shuffle=True, sample=1.0):
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

        min_net_size : int (default=3)
            If untagle=True then separate networks are not allowed to reconnect.
            That, unless their size is lower or equal than min_net_size.

        shuffle : boolean (default=True)
            Whether to shuffles the training data every epoch.

        sample : float (default=1.0)
            Sample a fraction of the training data for every epoch. Takes values
            between 0 and 1 (where 1 = the entire dataset).
        """

        self.units = []
        self.edges = []

        self.n_epochs = n_epochs
        self.sigma = sigma
        self.max_units = max_units
        self.eps_b = eps_b
        self.eps_n = eps_n
        self.max_edge_age = max_edge_age
        self.alpha = alpha
        self.d = d
        self.untangle = untangle
        self.min_net_size = min_net_size
        self.signal_counter = 0

        self.shuffle = shuffle
        self.sample = sample
        self.epsilon = .0   # anomaly threshold.
        

    def init_model(self, xs):
        n = len(xs) - 1
        a = Unit(xs[random.randint(0, n)])
        b = Unit(xs[random.randint(0, n)])

        a.neighbors.add(b)
        b.neighbors.add(a)
        self.units = [a, b]
        self.edges.append(Edge(a, b))


    def transform(self, xs, update_model_stats=False):
        if len(self.units) == 0:
            return None

        else:
            groups = {i: [] for i, _ in enumerate(self.units)}
            prototypes = np.array([u.prototype for u in self.units])
            labels = []

            for i, x in enumerate(xs):
                dists = np.linalg.norm(x - prototypes, axis=1)
                unit_id = np.argmin(dists)
                groups[unit_id].append(i)
                labels.append(unit_id)

            if update_model_stats:
                for k, v in groups.items():
                    unit = self.units[k]
                    points_in_unit = xs[v]
                    unit.count = len(points_in_unit)
                    unit.var = np.var(points_in_unit, axis=0)
                #     unit.cov = np.cov(points_in_unit, rowvar=False)

                # ps = [self.multinorm_pdf(x) for x in xs]
                # self.epsilon = np.min([p for p in ps if p > 0])

            return labels


    def norm_pdf(self, x):
        prototypes = np.array([u.prototype for u in self.units])
        dists = np.linalg.norm(x - prototypes, axis=1)
        unit_id = np.argmin(dists)
        unit = self.units[unit_id]
        mean, std = unit.prototype, np.sqrt(unit.var)
        p_x = [norm.pdf(v, u, s) for (v, u, s) in zip(x, mean, std)]
        
        return np.prod(p_x)


    def multinorm_pdf(self, x):
        prototypes = np.array([u.prototype for u in self.units])
        dists = np.linalg.norm(x - prototypes, axis=1)
        unit_id = np.argmin(dists)
        unit = self.units[unit_id]
        mean, cov = unit.prototype, unit.cov
        
        try:
            return multivariate_normal.pdf(x, mean, cov)
        except:
            return float('nan')


    def is_anomaly(self, x):
        return self.norm_pdf(x) < self.epsilon


    def fit(self, xs):
        for _ in range(0, self.n_epochs):
            self.partial_fit(xs)
            

    def fit_transform(self, xs):
        self.fit(xs)
        return self.transform(xs, update_model_stats=True)


    def partial_fit(self, xs):
        if len(self.units) == 0:
            self.init_model(xs)

        sigma = self.sigma
        alpha = self.alpha
        eps_b = self.eps_b
        eps_n = self.eps_n
        d = self.d
        max_edge_age = self.max_edge_age
        max_units = self.max_units

        signals = xs

        if self.shuffle or self.sample < 1.0:
            size = len(xs)
            n_samples = size
            if self.sample < 1.0: n_samples = int(size * self.sample)
            signals = xs[np.random.choice(size, n_samples)]

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

            elif not self.untangle or self.no_curling2(unit_a, unit_b):
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


    def net_size_compare(self, node, size):
        """
        Checks the size of the network a node belongs to, against the parameter size.
        Returns 1 if the size of the network is greater than "size", -1 if its lower,
        and 0 if equal.
        """

        open_nodes = node.neighbors
        closed_nodes = set()        
        closed_nodes.add(node)

        while len(open_nodes) > 0 and len(closed_nodes) <= size:
            closed_nodes = closed_nodes | open_nodes
            aux = set()
            for n in open_nodes:
                aux = aux | {ne for ne in n.neighbors if not ne in closed_nodes}
                
            open_nodes = aux
        
        n_nodes = len(closed_nodes)

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
        bridges = a.neighbors.intersection(b.neighbors)

        if len(bridges) == 2:
            # no curling if the two common neighbors are not connected.
            x, y = bridges
            return len(x.neighbors.intersection(y.neighbors)) == 0

        elif len(bridges) == 1:
            # no curling if there are less than 2 common neighbors between
            # 'a' and 'x', and between 'b' and 'x'.
            [x] = bridges
            xn = x.neighbors
            an = a.neighbors
            bn = b.neighbors

            return (len(an.intersection(xn)) < 2) and (len(bn.intersection(xn)) < 2)

        return False


    def no_curling2(self, a, b):
        """
        Another curl check version with three rules. No curling takes place if:
            - 'a' and 'b' have 2 neighbors in common, and those neighbors are not
            connected to each other.
            - 'a' and 'b' have 1 neighbor in common 'x', and there are no more than
            1 path from 'a' to 'x', and from 'b' to 'x'.
            - 'a' and 'b' belong to different networks.
        """

        bridges = a.neighbors.intersection(b.neighbors)
        n_bridges = len(bridges)

        if n_bridges == 2:
            # no curling if the two common neighbors are not connected.
            x, y = bridges
            return len(x.neighbors.intersection(y.neighbors)) == 0

        elif n_bridges == 1:
            max_steps = 5
            n_nodes, [bridge] = len(self.units), bridges
            xi = self.units.index(bridge)
            ai = self.units.index(a)
            bi = self.units.index(b)

            # construct edge matrix
            edge_mat = np.zeros(n_nodes**2).reshape(n_nodes, n_nodes)

            for e in self.edges:
                u1, u2 = self.units.index(e.source), self.units.index(e.target)
                edges[u1, u2] = 1
                edges[u2, u1] = 1

            # save edges of x (the bridge) without edges to a and b.
            x_edges = edges[xi].copy()
            x_edges[ai], x_edges[bi] = 0, 0
            x_edges = [i for i, e in enumerate(x_edges) if e == 1]
            # clear edges of x from the matrix
            edge_mat[xi] = 0
            edge_mat[:, xi] = 0
            
            # count paths from x to a and b, by restoring one edge of x at a time.
            n_paths = {ai: 0, bi: 0}

            for xe in x_edges:
                aux = edge_mat.copy()
                aux[xi, xe], aux[xe, xi] = 1, 1

                # to avoid long execution times, just check paths up to 5 steps away.
                for i in range(min(n_nodes-1, max_steps)):
                    aux = aux.dot(aux)

                if aux[xi, ai] != 0: n_paths[ai] += 1
                if aux[xi, bi] != 0: n_paths[bi] += 1

            return n_paths[ai] <= 1 and n_paths[bi] <= 1

        elif n_bridges == 0:
            return not self.exists_path(a, b)

        return False


    def save_vna(self, filename):
        """
        Saves graph (GNG model) to .vna format (which can then be loaded into, e.g., Gephi).
        """

        nodes = '*node data\nID name\n'
        nodes += '\n'.join([f'{i} {i}' for i, _ in enumerate(self.units)])

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
          label {i}
        ]
        """.format(i=i) for i, _ in enumerate(self.units)]

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
