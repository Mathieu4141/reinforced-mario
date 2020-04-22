"""
Use this script to compare the results of the fcn networks.
"""

import tensorflow as tf
from pandas import DataFrame

from constants import REPORT_DIR
from segmentation.dataset.batch_generator import BatchGenerator
from segmentation.dataset.sets import Set
from segmentation.fcn.evaluation import evaluate_accuracy
from segmentation.fcn.saved_fcn import SavedFCN
from utils.df_utils import format_df_column, df_to_latex_table
from utils.reproductibility import seed_all


def _evalute_models() -> DataFrame:
    rv = DataFrame(columns=["name", "accuracy", "fps"]).set_index("name")
    _vg = BatchGenerator(set_=Set.TEST, batch_size=64, randomize_before=False)
    for _fcn_name in (
        "fcn__f32-k3_f32-k3_s2__f64-k3_f64-k3_s2__d8",
        "fcn__f32-k3_s2__f64-k3_f64-k3_s2__d8",
        "fcn__f16-k3_s2__f32-k3_f32-k3_s2__d8",
    ):
        _fcn: SavedFCN = SavedFCN(len(_vg.color2id), _fcn_name)
        ac, t = evaluate_accuracy(_fcn, _vg)
        rv.loc[_fcn_name] = (ac, t)
        tf.compat.v1.reset_default_graph()
    return rv


def _make_latex_table(results_df: DataFrame):
    format_df_column(results_df, "accuracy", "{:.1%}")
    format_df_column(results_df, "fps", "{:.1f}")
    table = df_to_latex_table(results_df)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)
    (REPORT_DIR / "fcn_comparison_table.tex").write_text(table)


if __name__ == "__main__":
    seed_all()
    _df = _evalute_models()
    _make_latex_table(_df)
