from __future__ import print_function
from tf_importer.tf_importer.ops_bridge import OpsBridge


def get_unimplemented_ops(pb_path):
    # get required op
    with open(pb_path) as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

        required_ops = set()
        for line in lines:
            if line[:3] == 'op:':
                op_name = line.split(' ')[1][1:-1]
                required_ops.add(op_name)

    # get supported ops
    ob = OpsBridge()
    supported_ops = set([name for name in dir(ob)
                         if name[:1] != "_" and name not in ob.__dict__])

    # get unimplemented ops
    unimplemented_ops = required_ops - supported_ops
    return sorted(list(unimplemented_ops))


if __name__ == '__main__':
    for op in get_unimplemented_ops('../examples/graph.pb.txt'):
        print(op)
