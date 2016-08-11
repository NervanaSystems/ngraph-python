try:
    input = raw_input
except NameError:
    pass


class Debugger(object):
    def __init__(self):
        self.breakpoints = set()
        self.commands = []

    def debug(self, node):
        print 'In debugger...'
        while True:
            command = input()
            if command == 'continue':
                break
            elif command == 'node type':
                print type(node)
            elif command[:4] == 'exec':
                print eval(command[5:])
