from collections import defaultdict
from geon.backends.graph.ast import ComputationOp, AllocationOp, ElementWise, Function

def _random_colors(N, alpha=.5):
    from colorsys import hsv_to_rgb
    HSV = [[x*1.0/N, 0.5, 0.5] for x in range(N)]
    RGBA = [x + (alpha,) for x in map(lambda x: hsv_to_rgb(*x), HSV)]
    RGBA = [[int(y*255) for y in x] for x in RGBA]
    HEX = ["#{:02x}{:02x}{:02x}{:02x}".format(r,g,b,a) for r,g,b,a in RGBA]
    return HEX
    
class Digraph(object):
    
    def _graphviz(self, name=''):
        from graphviz import Digraph
        dot = Digraph(name)
        for node, nexts in self.successors.iteritems():
            dot.node(node.id, node.graph_label, node.style)
            for next in nexts:
                dot.node(next.id, next.graph_label, next.style)
                dot.edge(node.id, next.id)
        return dot
                
    @staticmethod
    def _invert(adjacency):
        result = {x: set() for x in adjacency.iterkeys()}
        for x, others in adjacency.iteritems():
            for y in others:
                result[y].add(x)
        return result

    def __init__(self, successors):
        self.successors = successors
        
    def render(self, fpath, view = True):
        self._graphviz().render(fpath, view=view)

    def view(self):
        self._graphviz().view()
        
    def topsort(self):
        visited = set()
        result = []
        predecessors = Digraph._invert(self.successors)
        counts = {a: len(b) for a, b in predecessors.iteritems()}
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
            self.successors[w] |= set()
            for v in w.inputs:
                self.successors[v].add(w)
                self._fill_successors({v})

    def __init__(self, outputs):
        super(DataFlowGraph, self).__init__(defaultdict(set))
        self._fill_successors(outputs)
        self.outputs = outputs


class KernelFlowGraph(DataFlowGraph):

    @staticmethod
    def _fusible(op1, op2):
        if not isinstance(op1, ComputationOp) or not isinstance(op2, ComputationOp):
            return False
        if isinstance(op1, ElementWise) and isinstance(op2, ElementWise):
            return True
        return False
        
    def _graphviz(self, name=''):
        predecessors = Digraph._invert(self.successors)
        from graphviz import Digraph as gvDigraph
        dot = gvDigraph(name, graph_attr={'compound':'true', 'nodesep':'.5', 'ranksep':'.5'})
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
        return dot
        
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
            if x!=w:
                worklist |= {y for y in self.successors[x] if w in path_from[y]}
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
        clusters = {x: Function(y) if isinstance(x, ComputationOp) else x for x, y in clusters.iteritems()}
        self.successors = {clusters[a]: {clusters[b] for b in lst} for a, lst in successors.iteritems()}
        
        #Saves dataflow for visualization
        self.dataflow = dataflow
        

    def liveness(self):
        order = self.topsort()
        #Initialize
        liveness = dict((op,set()) for op in order)
        keeps = {x for x in self.successors if isinstance(x, AllocationOp) and x.keep}
        liveness[order[-1]] = set(self.outputs) | keeps
        #Update
        for current, previous in reversed(zip(order[1:], order[:-1])):
            liveness[previous] = set(current.use) | (liveness[current] - set(current.defs))
        return liveness
    

