from geon.backends.graph.defmodimp import ComputedTensor, ElementWise, Function, Variable, input

def _random_colors(N, alpha=.5):
    from colorsys import hsv_to_rgb
    HSV = [[x*1.0/N, 0.5, 0.5] for x in range(N)]
    RGBA = [x + (alpha,) for x in map(lambda x: hsv_to_rgb(*x), HSV)]
    RGBA = [[int(y*255) for y in x] for x in RGBA]
    HEX = ["#{:02x}{:02x}{:02x}{:02x}".format(r,g,b,a) for r,g,b,a in RGBA]
    return HEX
    
class Digraph(object):
    
    def _graphviz(self, name='', getinfo = None):
        from graphviz import Digraph
        dot = Digraph(name)
        for node, nexts in self.successors.iteritems():
            dot.node(node.id, node.graph_label, node.style)
            if node.graph_label is None:
                print type(node)
            for next in nexts:
                dot.node(next.id, next.graph_label, next.style)
                dot.edge(node.id, next.id)
        return dot
                
    @staticmethod
    def _invert(adjacency):
        result = dict((x, set()) for x in adjacency.keys())
        for x, others in adjacency.iteritems():
            for y in others:
                result.setdefault(y, set()).add(x)
        return result

    def __init__(self, successors):
        self.successors = successors
        
    def visualize(self, fname):
        self._graphviz().render('/tmp/{}.gv'.format(fname), view=True)

    def topsort(self):
        visited = set()
        result = []
        predecessors = Digraph._invert(self.successors)
        counts = dict((a, len(b)) for a, b in predecessors.iteritems())
        queue = [node for node,count in counts.iteritems() if count == 0]
        while queue:
            #Dequeue node with all dependency satisfied
            current = queue.pop(0)
            result.append(current)
            #Decrement neighbors dependency count
            for nxt in self.successors[current]:
                counts[nxt] -= 1
                if counts[nxt] == 0:
                    queue.append(nxt)
        return result

class DataFlowGraph(Digraph):
    
    def _fill_successors(self, outputs):
        for w in outputs:
            self.successors.setdefault(w, set())
            use = set(w.args)
            for v in use:
                self.successors.setdefault(v, set()).add(w)
            self._fill_successors(use)

    def __init__(self, outputs):
        super(DataFlowGraph, self).__init__(dict())
        self._fill_successors(outputs)
        self.outputs = outputs


class KernelFlowGraph(DataFlowGraph):

    @staticmethod
    def _fusible(op1, op2):
        if not isinstance(op1, ComputedTensor) or not isinstance(op2, ComputedTensor):
            return False
        if isinstance(op1, ElementWise) and isinstance(op2, ElementWise):
            return True
        return False
    
    def _compute_paths(self):
        path_from, bad_path_from = dict(), dict()
        order = self.topsort()
        for v in reversed(order):
            path_from[v] = {v}
            bad_path_from[v] = set()
            for w in self.successors[v]:
                path_from[v] |= path_from[w]
                bad_path_from[v] |= path_from[w] if not KernelFlowGraph._fusible(v, w) else bad_path_from[w]
        return path_from, bad_path_from

    def between(self, v, w, path_from):
        vertices = set()
        #Initialize worklists to all successors of v who can reach w
        worklist = {w}
        worklist |= {x for x in self.successors[v] if w in path_from[x]}
        while worklist:
            #Update worklist
            x = worklist.pop()
            if x!=w: worklist |= {y for y in self.successors[x] if w in path_from[y]}
            #Add vertices
            vertices |= {x}
        return vertices

    def transfer_edges(self, v, w, dct):
        dct[v] |= dct.pop(w, set()) - {v}
        for node, connected in dct.iteritems():
            if w in connected:
                connected.remove(w)
                if node != v:
                    connected.add(v)
                    
    def __init__(self, dataflow):
        #Extracts clusters
        super(KernelFlowGraph, self).__init__(dataflow.outputs)
        successors = self.successors
        
        path_from, bad_path_from = self._compute_paths()
        edges = {(a, b) for a, _ in successors.iteritems() for b in _}
        clusters = dict((x,{x}) for e in edges for x in e)
        while edges:
            #Pop edges and adjusts order if necessary
            v, w = edges.pop()
            #Cannot be fused
            if w in bad_path_from[v]:
                continue
            #Merge vertices between v and w
            to_merge = self.between(v, w, path_from)
            for x in to_merge:
                clusters[v] |= clusters.pop(x)
                self.transfer_edges(v, x, successors)
                self.transfer_edges(v, x, path_from)
                self.transfer_edges(v, x, bad_path_from)
            edges = {(a, b) for a, _ in successors.iteritems() for b in _}
        #Creates adjacency list for each cluster
        extract_subgraph = lambda R: dict((a, b & R) for a, b in dataflow.successors.iteritems() if a in R)
        clusters = {x: extract_subgraph(y) for x, y in clusters.iteritems()}
        #Creates final adjacency list
        clusters = {x: Function(y) if isinstance(x, ComputedTensor) else x for x, y in clusters.iteritems()}
        self.successors = {clusters[a]: {clusters[b] for b in lst} for a, lst in successors.iteritems()}
        
        #Saves dataflow for visualization
        self.dataflow = dataflow
        
    def visualize(self, fname):
        predecessors = Digraph._invert(self.successors)
        from graphviz import Digraph as gvDigraph
        dot = gvDigraph(graph_attr={'compound':'true', 'nodesep':'.5', 'ranksep':'.5'})
        leaves = {x for x, y in predecessors.iteritems() if len(y)==0}
        subgs = {x: x.ops._graphviz('cluster_{}'.format(x.id)) for x in self.successors if isinstance(x, Function)}
        #Subgraphs
        for x, sg in subgs.iteritems():
            sg.body.append('color=gray')
            sg.body.append('label={}'.format(x.id))
            dot.subgraph(sg)
        for x in leaves:
            dot.node(x.id, x.graph_label, x.style)
        #Edges
        edges = {(a, b) for a, _ in self.successors.iteritems() for b in _}
        sorts = {x: x.ops.topsort() for x in self.successors if isinstance(x, Function)}
        firsts = {x: sorts[x][0] if isinstance(x, Function) else x for x in self.successors}
        lasts = {x: sorts[x][-1] if isinstance(x, Function) else x for x in self.successors}  
        for a, b in edges:
            kw = {}
            if isinstance(a, Function): kw['ltail'] = 'cluster_{}'.format(a.id)
            if isinstance(b, Function): kw['lhead'] = 'cluster_{}'.format(b.id)
            edge = dot.edge(lasts[a].id, firsts[b].id, **kw)
        dot.render('/tmp/{}.gv'.format(fname), view=True)


    def liveness(self):
        order = self.topsort()
        #Initialize
        liveness = dict((op,set()) for op in order)
        keeps = {x for x in self.successors if isinstance(x, Variable) and x.keep}
        liveness[order[-1]] = set(self.outputs) | keeps
        #Update
        for current, previous in reversed(zip(order[1:], order[:-1])):
            liveness[previous] = set(current.use) | (liveness[current] - set(current.defs))
        return liveness
        
class UndirectedGraph(object):

    def __init__(self, successors):
        self.successors = successors
        
    def visualize(self, fname):
        from graphviz import Graph
        dot = Graph()
        processed = set()
        for na, _ in self.successors.iteritems():
            dot.node(na.label, na.label, na.style)
            for nb in _:
                dot.node(nb.label, nb.label, nb.style)
                if (nb, na) not in processed:
                    dot.edge(na.label, nb.label)
                    processed.add((na,nb))
        dot.render('/tmp/{}.gv'.format(fname), view=True)

    def color(self):
        neighbors_map = dict(self.successors)
        for node, preds in Digraph._invert(self.successors).iteritems():
            neighbors_map[node] |= preds
        degrees = [len(neighbors) for neighbors in neighbors_map.itervalues()]
        maxdegree = max(degrees)
        colors = set(range(maxdegree+1))
        for na, neighbors in neighbors_map.iteritems():
            na.color = min(colors - {nb.color for nb in neighbors})
        cmap = _random_colors(len({x.color for x in neighbors_map}), .5)
        for na in neighbors_map:
            na.style = {'style':'filled', 'fillcolor': cmap[na.color]}

    
class InterferenceGraph(UndirectedGraph):
    
    def __init__(self, kernelflow):
        from itertools import combinations
        succs = dict()
        lives = kernelflow.liveness()
        succs = {x: set() for y in lives.itervalues() for x in y}

        for lst in lives.itervalues():
            for u, v in combinations(lst, 2):
                succs.setdefault(u, set()).add(v)
        super(InterferenceGraph, self).__init__(succs)
    

