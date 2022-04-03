class VertexGroup(object):
    def __init__(self, *indicies, normals=[]):
        self.indicies = indicies
        self.vertex_normals = normals
