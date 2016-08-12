from geon.op_graph import nodes


def test_bytes_tags():
    n = nodes.Node(tags=b'abc')
    assert len(n.tags) == 1


def test_unicode_tags():
    n = nodes.Node(tags=u'abc')
    assert len(n.tags) == 1
