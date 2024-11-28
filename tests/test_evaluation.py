import pandas as pd
from more_itertools import one

from tests.utils import timeseries
from tsgraph.evaluation import get_execution_plan, run_execution_plan
from tsgraph.nodes.core import df_node, output_node
from tsgraph.nodes.maths import add, mul
from tsgraph.nodes.utils import lag


def test_get_execution_plan():
    df_n = df_node(timeseries([1, 2, 3, 4]))
    lag_n = lag(df_n, 1)
    mul_n = mul(lag_n, -1)
    add_n = add(df_n, mul_n)
    align_n = one(n for n in add_n.parents if 'align' in str(n))
    # df appears multiple times and lag_n is stateful
    t1 = pd.Timestamp('2000-01-01')
    t2 = pd.Timestamp('2000-01-02')
    t3 = pd.Timestamp('2000-01-03')

    plan = get_execution_plan([add_n], t1, t2)
    expected = [(df_n, (None, t2), set()),
                (lag_n, (None, t2), set()),
                (mul_n, (None, t2), {lag_n}),
                (align_n, (t1, t2), {mul_n}),
                (add_n, (t1, t2), {df_n, align_n})]
    assert plan == expected

    plan = get_execution_plan([add_n], None, t1)
    expected = [(df_n, (None, t1), set()),
                (lag_n, (None, t1), set()),
                (mul_n, (None, t1), {lag_n}),
                (align_n, (None, t1), {mul_n}),
                (add_n, (None, t1), {df_n, align_n})]
    assert plan == expected

    # Now we actually execute the plan.
    run_execution_plan(plan)

    plan = get_execution_plan([add_n], t1, t2)
    expected = [(df_n, (t1, t2), set()),
                (lag_n, (t1, t2), set()),
                (mul_n, (t1, t2), {lag_n}),
                (align_n, (t1, t2), {mul_n}),
                (add_n, (t1, t2), {df_n, align_n})]
    assert plan == expected

    # But if we try t2 -> t3, it should have to regenerate the align fully.
    plan = get_execution_plan([add_n], t2, t3)
    expected = [(df_n, (None, t3), set()),
                (lag_n, (None, t3), set()),
                (mul_n, (None, t3), {lag_n}),
                (align_n, (t2, t3), {mul_n}),
                (add_n, (t2, t3), {df_n, align_n})]
    assert plan == expected

def test_get_execution_plan_output_node():
    df_n = df_node(timeseries([1, 2, 3, 4]))
    output_n = output_node(df_n)
    add2_n = add(output_n, 1)

    t1 = pd.Timestamp('2000-01-01')
    t2 = pd.Timestamp('2000-01-02')
    t3 = pd.Timestamp('2000-01-03')

    plan = get_execution_plan([add2_n], None, t1)
    expected = [(df_n, (None, t1), set()),
                (output_n, (None, t1), {df_n}),
                (add2_n, (None, t1), {output_n})]
    assert plan == expected

    plan = get_execution_plan([add2_n], t1, t2)
    expected = [(df_n, (None, t2), set()),
                (output_n, (t1, t2), {df_n}),
                (add2_n, (t1, t2), {output_n})]
    assert plan == expected

    # Run the plan.
    run_execution_plan(plan)

    plan = get_execution_plan([add2_n], t1, t2)
    expected = [(output_n, (t1, t2), set()),
                (add2_n, (t1, t2), {output_n})]
    assert plan == expected

    plan = get_execution_plan([add2_n], None, t2)
    expected = [(output_n, (None, t2), set()),
                (add2_n, (None, t2), {output_n})]
    assert plan == expected

    plan = get_execution_plan([add2_n], None, t3)
    expected = [(df_n, (t2, t3), set()),
                (output_n, (None, t3), {df_n}),
                (add2_n, (None, t3), {output_n})]
    assert plan == expected


