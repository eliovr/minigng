import numpy as np
import random


class Unit:
    def __init__(self, prototype, error=.0):
        self.prototype = prototype
        self.error = error
        self.neighbors = set()

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
            untangle=False, min_net_size=3, shuffle=True):
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
        """

        self.units = []
        self.edges = []

        self.n_epochs = n_epochs
        self.shuffle = shuffle
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


    def init_model(self, xs):
        n = len(xs) - 1
        a = Unit(xs[random.randint(0, n)])
        b = Unit(xs[random.randint(0, n)])

        a.neighbors.add(b)
        b.neighbors.add(a)
        self.units = [a, b]
        self.edges.append(Edge(a, b))


    def fit(self, xs):
        for _ in range(0, self.n_epochs):
            self.partial_fit(xs)


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
        if self.shuffle:
            size = len(xs)
            signals = xs[np.random.choice(np.arange(size), size)]

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
            _edges = []

            for e in self.edges:
                # 3. Increment the age of all edges emanating from S1.
                if e.connects_unit(unit_a):
                    e.age += 1

                    if e.connects_unit(unit_b):
                        ab_edge = e

                # 7. Remove edges with an age larger than maxAge. If this results in
                # points having no emanating edges, remove them as well.
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


    def save_gml(self, filename):
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
