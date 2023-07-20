from deepcrunch.utils.dot_dict import DotDict, deep_get, deep_set

def test_dot_dict():
    d = DotDict({'a': 1, 'b': {'c': 2}})
    assert d.a == 1
    assert d.b.c == 2 # pyright: ignore[reportOptionalMemberAccess]
    d.b.c = 3 # pyright: ignore[reportOptionalMemberAccess]
    assert d.b.c == 3 # pyright: ignore[reportOptionalMemberAccess]

def test_deep_get():
    d = {'a': {'b': {'c': 1}}}
    assert deep_get(d, 'a.b.c') == 1
    assert deep_get(d, 'a.b.d') is None
    assert deep_get(d, 'a.b.d', default=2) == 2

def test_deep_set():
    d = {}
    deep_set(d, 'a.b.c', 1)
    assert d == {'a': {'b': {'c': 1}}}
    deep_set(d, 'a.b.d', 2)
    assert d == {'a': {'b': {'c': 1, 'd': 2}}}
    deep_set(d, 'a.b', {'e': 3})
    assert d == {'a': {'b': {'e': 3}}}