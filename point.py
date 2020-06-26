class Point:

    def __init__(self, idx, vec, dist, preceding=None, following=None):
        self.idx = idx
        self.preceding = preceding
        self.following = following
        self.vec = vec
        self.dist = dist

    def __repr__(self):
        return "Point(idx: {}, vector: {}, dist: {}".format(self.idx, self.vec, self.dist)

    def __str__(self):
        return "idx: {}, vector: {}, dist: {}".format(self.idx, self.vec, self.dist)
