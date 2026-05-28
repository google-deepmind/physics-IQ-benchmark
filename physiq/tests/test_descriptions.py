import pandas as pd

from dataset import DESCRIPTIONS_PATH, Benchmark

_DESCRIPTIONS_CSV = DESCRIPTIONS_PATH.parent / "descriptions.csv"


def test_yaml_generates_matching_descriptions_csv():
    csv_gen = Benchmark.from_yaml(DESCRIPTIONS_PATH).build_original_descriptions()
    csv_disk = pd.read_csv(_DESCRIPTIONS_CSV)
    pd.testing.assert_frame_equal(
        csv_gen.reset_index(drop=True),
        csv_disk.reset_index(drop=True),
    )
