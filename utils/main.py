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


def normalize(arr, t_min, t_max):
    """Normalize a list given a range."""
    norm_arr = []
    diff = t_max - t_min

    arr_max = max(arr)
    arr_min = min(arr)

    diff_arr = arr_max - arr_min

    if diff_arr == 0:
        return [0] * len(arr)

    for i in arr:
        temp = round((((i - arr_min) * diff) / diff_arr) + t_min, 4)
        norm_arr.append(temp)
    return norm_arr
