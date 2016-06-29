import weakref

class Node(object):

    def __init__(self, args=(), **kargs):
        if kargs:
            pass
        super(Node, self).__init__(**kargs)
        self.users = weakref.WeakSet()
        self.__inputs = ()
        self.inputs = args

    @property
    def inputs(self):
        """All the inputs to this node"""
        return self.__inputs

    @inputs.setter
    def inputs(self, args):
        """
        Replace old inputs with new inputs, adjusting backpointers as needed.
        :param args: New arguments
        :return:
        """
        for arg in self.__inputs:
            arg.users.remove(self)
        self.__inputs = self.as_nodes(args)
        for arg in self.__inputs:
            arg.users.add(self)

    def as_nodes(self, args):
        return tuple(self.as_node(arg) for arg in args)

    def as_node(self, arg):
        """Override to convert an object to a node"""
        return arg

    @staticmethod
    def visit_input_closure(root, fun):
        """
        Bottom-up traversal of root and their inputs
        :param root: root set of nodes to visit
        :param fun: Function to call on each visited node
        """
        visited = set()

        def visit(node):
            if node not in visited:
                for n in node.inputs:
                    visit(n)
                fun(node)
                visited.add(node)

        for node in root:
            visit(node)

    @staticmethod
    def visit_output_closure(root, fun):
        """
        Top-down traversal of root and closure of nodes using root as input.
        :param root:  root set of nodes to visit
        :param fun: Function to call on each visited node
        :return:
        """
        visited = set()

        def visit(node):
            if node not in visited:
                for n in node.users:
                    visit(n)
                fun(node)
                visited.add(node)

        for node in root:
            visit(node)




