import csv
import json
import math

import pytest

from isdynamic.release.tables import controls_table
from isdynamic.viz.analyze_backbone_probe_diagnostic import normalized_gain, pearson, spearman, summarize


def test_ng_accounts_for_class_imbalance():
    ng, baseline, positive_rate = normalized_gain({"tp": 85, "fn": 5, "fp": 2, "tn": 8, "accuracy": 0.93})
    assert baseline == 0.9 and positive_rate == 0.9
    assert ng == pytest.approx(0.3)
    ng, _, _ = normalized_gain({"tp": 0, "fn": 90, "fp": 10, "tn": 0, "accuracy": 0.0})
    assert ng < 0


@pytest.mark.parametrize("counts", [(0,0,0,0), (5,0,0,0)])
def test_ng_rejects_undefined_denominator(counts):
    with pytest.raises(ValueError):
        normalized_gain(dict(zip(("tp","fn","fp","tn"), counts), accuracy=1.0))


def test_association_and_population_dispersion():
    assert spearman([1,2,3], [30,20,10]) == pytest.approx(-1)
    assert pearson([1,2,3], [7,9,11]) == pytest.approx(1)
    assert spearman([1,1,2,3], [2,2,4,6]) == pytest.approx(1)
    assert math.isnan(pearson([1,1,1], [1,2,3]))
    summary = summarize([1,2,3])
    assert summary["mean"] == 2
    assert summary["population_std"] == pytest.approx(math.sqrt(2/3))


def test_control_table_averages_all_blocks_and_records_source(tmp_path):
    experiment = tmp_path / "controls/pi3x/correct_rank_8"
    for index, metrics in enumerate((
        {"tp":70,"fn":10,"fp":5,"tn":15,"accuracy":0.85},
        {"tp":60,"fn":20,"fp":10,"tn":10,"accuracy":0.70},
    )):
        path = experiment / f"pair_lowrank_quadratic/inter_frame_block/inter_frame_block_{index}/metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"best_val": metrics}))
    controls_table(tmp_path)
    with (tmp_path / "tables/probe_controls.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert int(row["Blocks"]) == 2
    assert float(row["Pair accuracy (%)"]) == pytest.approx(77.5)
    assert float(row["Gain (pp)"]) == pytest.approx(-2.5)
    assert row["Source"] == str(experiment)
