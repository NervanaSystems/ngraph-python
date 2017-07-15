from ngraph.util.names import ScopedNameableValue, name_scope


def test_nested_namescope():
    """
    Ops under multiple namescopes should have nested names. Namescope names don't nest
    so that they can be reused in different nestings.
    """
    with name_scope("scope1") as scope1:
        assert scope1.name == "scope1"
        val1 = ScopedNameableValue("val1")
        assert val1.name == "scope1/val1"
        with name_scope("scope2") as scope2:
            assert scope2.name == "scope2"
            val2 = ScopedNameableValue("val2")
            assert val2.name == "scope1/scope2/val2"


def test_scope_reuse():
    """
    Namescopes are only reused if explicitly requested. Otherwise, a uniquely named will
    be created.
    """

    with name_scope("scope") as scope1:
        val1 = ScopedNameableValue("val1")

    with name_scope("scope", reuse_scope=True) as scope2:
        val2 = ScopedNameableValue("val2")

    with name_scope("scope", reuse_scope=False) as scope3:
        val3 = ScopedNameableValue("val3")

    assert scope1 is scope2
    assert scope1 is not scope3
    assert val1.name == "scope/val1"
    assert val2.name == "scope/val2"
    assert val3.name != "scope/val3"
