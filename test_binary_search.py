from simulate_minibatch_binary_search import binary_search
from random import shuffle


def test_binary_search():
    xs = list(range(100))
    shuffle(xs)

    max_total = sum(xs)
    # print(max_total)

    for t in range(1, max_total):
        result_idx, sum_now = binary_search(xs, 0, t)
        assert sum_now <= t, "idx {idx}, sum_now {sum_now}, t {t}".format(idx=result_idx, sum_now=sum_now, t=t)
        assert sum_now + xs[result_idx + 1] > t, "idx {idx}, sum_now {sum_now}, t {t}".format(
            idx=result_idx, sum_now=sum_now, t=t)


def test_binary_search_many_length():
    for xs_len in range(95, 100):
        xs = list(range(xs_len))
        shuffle(xs)

        max_total = sum(xs)
        # print(max_total)

        for t in range(1, max_total):
            result_idx, sum_now = binary_search(xs, 0, t)
            assert sum_now <= t, "idx {idx}, sum_now {sum_now}, t {t}".format(idx=result_idx, sum_now=sum_now, t=t)
            assert sum_now + xs[result_idx + 1] > t, "idx {idx}, sum_now {sum_now}, t {t}".format(
                idx=result_idx, sum_now=sum_now, t=t)


def test_binary_search_sum_cur():
    xs = list(range(100))
    # shuffle(xs)

    max_total = sum(xs)
    # print(max_total)

    sum_cur = 500

    for t in range(1, max_total):
        idx, sum_incl_idx = binary_search(xs, sum_cur, t + sum_cur)
        sum_incl_idx_within_xs = sum(xs[0:idx + 1])
        assert sum_incl_idx == sum_incl_idx_within_xs + sum_cur
        assert sum_incl_idx <= t + sum_cur, "idx {idx}, sum_now {sum_now}, t {t}".format(idx=idx, sum_now=sum_incl_idx, t=t)
        assert sum_incl_idx + xs[idx + 1] > t + sum_cur, "idx {idx}, sum_now {sum_now}, t {t}".format(idx=idx,
                                                                                                         sum_now=sum_incl_idx,
                                                                                                         t=t)
def test_binary_search_last_item():
    xs = list(range(10))
    max_total = sum(xs)
    sum_cur = 0
    t = max_total
    idx, sum_incl_idx = binary_search(xs, sum_cur, t + sum_cur)
    sum_incl_idx_within_xs = sum(xs[0:idx + 1])
    assert sum_incl_idx == sum_incl_idx_within_xs + sum_cur
    assert idx == 9

