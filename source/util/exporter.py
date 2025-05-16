import os


class Exporter:
    results = os.path.join('results')
    quadrangulated = os.path.join('results', 'quadrangulated')
    torched = os.path.join('results', 'torched')
    binaries = os.path.join('results', 'binaries')
    loss = os.path.join('results', 'loss')
    stl = os.path.join('results', 'stl')
    tgt = os.path.join('results', 'normalized_target')
    meta = os.path.join('results', 'meta')
    

    @staticmethod
    def dirfill():
        os.makedirs(Exporter.results, exist_ok=True)
        os.makedirs(Exporter.quadrangulated, exist_ok=True)
        os.makedirs(Exporter.torched, exist_ok=True)
        os.makedirs(Exporter.binaries, exist_ok=True)
        os.makedirs(Exporter.loss, exist_ok=True)
        os.makedirs(Exporter.stl, exist_ok=True)
        os.makedirs(Exporter.tgt, exist_ok=True)
        os.makedirs(Exporter.meta, exist_ok=True)

    def __init__(self, mesh: str, lod: int, features: int):
        Exporter.dirfill()

        self.prefix = os.path.basename(mesh)
        self.prefix = self.prefix.split('.')[0]
        self.basename = self.prefix + f'-lod{lod}-f{features}'

    def partitioned(self):
        return os.path.join(Exporter.quadrangulated, self.basename + '.obj')

    def pytorch(self):
        return os.path.join(Exporter.torched, self.basename + '.pt')

    def binary(self):
        return os.path.join(Exporter.binaries, self.basename + '.bin')

    def plot(self):
        return os.path.join(Exporter.loss, self.basename + '.pdf')

    def mesh(self):
        return os.path.join(Exporter.stl, self.basename + '.stl')

    def metadata(self):
        return os.path.join(Exporter.meta, self.basename + '.json')
    
    def target(self):
        return os.path.join(Exporter.tgt, self.basename + '.stl')
