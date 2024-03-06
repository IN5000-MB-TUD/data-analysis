def sorted_simple_dict(d):
    """Sort simple dictionary by keys"""
    return {k: v for k, v in sorted(d.items())}


def sorted_once_nested_dict(d, key):
    """Sort nested dictionary by inner key and remove empty keys"""
    return {
        k: sorted_simple_dict(v)
        for k, v in sorted(d.items(), key=lambda x: x[1][key])
        if k
    }


def nearest(items, pivot):
    """Find nearest index and item in a list to the given item"""
    return min(enumerate(items), key=lambda x: abs(x[1] - pivot))
