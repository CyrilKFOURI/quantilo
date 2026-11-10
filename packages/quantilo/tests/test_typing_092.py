"""092: real-data typing defects (docs/features/092-corrections-multidomaine-import-types)."""

from pathlib import Path

import duckdb
import pytest

from quantilo.typing.engine import detect_columns
from quantilo.typing.knowledge_base import head_word, load
from quantilo.typing.parsers import typed_sql
from quantilo.typing.summary import summarize
from quantilo.typing.types import ColumnDetection
from tests.factories import write_csv
from tests.test_typing import detect, stored


def table(
    tmp_path: Path, header: list[str], rows: list[list[object]]
) -> dict[str, ColumnDetection]:
    lines = [",".join(f'"{name}"' for name in header)]
    lines += [",".join(f'"{value}"' if value != "" else "" for value in row) for row in rows]
    return detect(write_csv(tmp_path / "t.csv", "\n".join(lines) + "\n"), tmp_path)


def coded(detection: ColumnDetection, code: str) -> bool:
    return any(f"({code})" in reason for reason in detection.reasons)


def invalid_count(tmp_path: Path, detection: ColumnDetection) -> int:
    source = stored(tmp_path / "t.csv", tmp_path)
    summary = summarize(
        duckdb.connect(), source, detection.name, detection.type, detection.parser, detection.order
    )
    return summary["invalid_count"]


class TestNoValueLost:
    """R-TYP-01 / R-TYP-02 (TYP-1, TYP-2, TYP-4)."""

    def test_codes_with_letters_stay_text(self, tmp_path: Path) -> None:
        # French departments: 2A / 2B (Corsica) among numbers
        deps = [str(10 + i % 80) for i in range(980)] + ["2A", "2B"] * 10
        detections = table(tmp_path, ["dep", "n"], [[dep, i] for i, dep in enumerate(deps)])

        dep = detections["dep"]
        assert dep.type == "categorical_nominal"
        assert dep.parser.kind == "text"
        assert coded(dep, "R-TYP-01")
        assert any("2A" in reason for reason in dep.reasons)
        assert invalid_count(tmp_path, dep) == 0

    def test_icd9_codes_stay_codes(self, tmp_path: Path) -> None:
        codes = [f"{250 + i % 300}.{i % 9}" if i % 3 else str(400 + i % 200) for i in range(960)]
        codes += [f"V{50 + i % 9}" for i in range(20)] + [f"E9{i % 20:02d}" for i in range(20)]
        detections = table(tmp_path, ["diag_1", "n"], [[c, i] for i, c in enumerate(codes)])

        assert detections["diag_1"].parser.kind == "text"
        assert detections["diag_1"].type in ("categorical_nominal", "identifier")
        assert coded(detections["diag_1"], "R-TYP-01")

    def test_a_few_unreadable_values_still_read_as_numbers(self, tmp_path: Path) -> None:
        # 0.2 % unreadable: below the 0.5 % limit, the measure stays a measure (shown as invalid)
        values = [f"{i * 1.5:.1f}" for i in range(998)] + ["n.c.", "abc"]
        detections = table(tmp_path, ["weight_kg", "n"], [[v, i] for i, v in enumerate(values)])

        assert detections["weight_kg"].type == "numeric_continuous"

    def test_words_in_place_of_numbers_are_read_as_missing(self, tmp_path: Path) -> None:
        # 2.5 % of words without a digit: annotations, not codes (R-TYP-02, not R-TYP-01)
        words = ["see note", "heavy", "?kg"]
        values = [words[i % 3] if i % 40 == 0 else f"{i * 0.7:.1f}" for i in range(400)]
        detections = table(tmp_path, ["weight", "n"], [[v, i] for i, v in enumerate(values)])

        assert detections["weight"].type == "numeric_continuous"
        assert coded(detections["weight"], "R-TYP-02")

    def test_a_suppression_marker_is_read_as_missing(self, tmp_path: Path) -> None:
        # SAT scores: "s" = suppressed (12 %)
        scores = ["s" if i % 8 == 0 else str(350 + i % 250) for i in range(480)]
        detections = table(tmp_path, ["math_avg", "n"], [[s, i] for i, s in enumerate(scores)])

        math = detections["math_avg"]
        assert math.type == "numeric_continuous"
        assert math.parser.kind == "number"
        assert coded(math, "R-TYP-02")
        assert invalid_count(tmp_path, math) == 60


class TestCodes:
    """R-TYP-03 / 04 / 05 (TYP-1, TYP-3)."""

    def test_leading_zeros_make_a_code(self, tmp_path: Path) -> None:
        communes = [f"{1001 + i * 7:05d}" for i in range(300)]
        detections = table(
            tmp_path, ["code_commune", "n"], [[c, i % 4] for i, c in enumerate(communes)]
        )

        commune = detections["code_commune"]
        assert commune.parser.kind == "text"
        assert commune.type in ("identifier", "categorical_nominal")
        assert coded(commune, "R-TYP-03")

    def test_leading_zeros_win_over_a_few_words(self, tmp_path: Path) -> None:
        # a license number "0616847" with a few "N/C"-like words: still a code kept as text
        values = ["NONE" if i % 300 == 0 else f"{600_000 + i * 13:07d}" for i in range(600)]
        detections = table(tmp_path, ["license__", "n"], [[v, i] for i, v in enumerate(values)])

        assert detections["license__"].parser.kind == "text"

    def test_zero_padded_months_stay_months(self, tmp_path: Path) -> None:
        months = [f"{i % 12 + 1:02d}" for i in range(300)]
        detections = table(tmp_path, ["mois", "n"], [[m, i] for i, m in enumerate(months)])

        assert detections["mois"].type == "numeric_discrete"
        assert detections["mois"].role == "month"

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("patient_nbr", "categorical_nominal"),  # repeated patients
            ("unique_key", "identifier"),
            ("permit_si_no", "identifier"),
            ("MS SubClass", "categorical_nominal"),
            ("council_district", "categorical_nominal"),
            ("Code de la catégorie socio-professionnelle", "categorical_nominal"),
            ("Mother's occupation", "categorical_nominal"),
        ],
    )
    def test_a_code_name_on_whole_numbers(self, tmp_path: Path, name: str, expected: str) -> None:
        unique = expected == "identifier"
        values = [
            1_000_000 + i * 37 if unique else [20, 30, 50, 60, 120, 160, 190][i % 7] * (1 + i % 3)
            for i in range(400)
        ]
        detections = table(tmp_path, [name, "x"], [[v, i * 0.5] for i, v in enumerate(values)])

        assert detections[name].type == expected
        assert detections[name].parser.kind == "text"
        assert coded(detections[name], "R-TYP-04") or detections[name].role is not None

    def test_a_unit_in_the_name_says_a_measure(self, tmp_path: Path) -> None:
        # King County houses: "sqft_lot" is the lot's area, not a lot number
        values = [5_000 + (i * 37) % 9_000 for i in range(400)]
        detections = table(tmp_path, ["sqft_lot", "x"], [[v, i] for i, v in enumerate(values)])

        assert detections["sqft_lot"].type == "numeric_continuous"

    def test_a_count_name_stays_a_count(self, tmp_path: Path) -> None:
        values = [i % 90 for i in range(400)]
        detections = table(
            tmp_path, ["num_lab_procedures", "x"], [[v, i] for i, v in enumerate(values)]
        )

        assert detections["num_lab_procedures"].type == "numeric_continuous"

    def test_patient_nbr_is_a_customer_key(self, tmp_path: Path) -> None:
        values = [8_000_000 + (i % 150) * 1_237 for i in range(400)]
        detections = table(tmp_path, ["patient_nbr", "x"], [[v, i] for i, v in enumerate(values)])

        assert detections["patient_nbr"].role == "customer_key"
        assert detections["patient_nbr"].type == "categorical_nominal"

    def test_unique_long_whole_numbers_are_identifiers(self, tmp_path: Path) -> None:
        values = [202400000001 + i for i in range(400)]
        detections = table(tmp_path, ["Num_Acc", "x"], [[v, i] for i, v in enumerate(values)])

        assert detections["Num_Acc"].type == "identifier"
        assert coded(detections["Num_Acc"], "R-TYP-05")

    def test_fixed_length_long_codes_are_codes(self, tmp_path: Path) -> None:
        values = [1_000_000_000 * (1 + i % 5) + (i % 120) * 10_001 for i in range(400)]
        detections = table(tmp_path, ["bbl", "x"], [[v, i] for i, v in enumerate(values)])

        assert detections["bbl"].type == "categorical_nominal"
        assert coded(detections["bbl"], "R-TYP-05")

    def test_measures_of_varying_length_stay_measures(self, tmp_path: Path) -> None:
        values = [round(85_000 * 1.013**i) for i in range(400)]
        detections = table(tmp_path, ["SalePrice", "x"], [[v, i] for i, v in enumerate(values)])

        assert detections["SalePrice"].type == "numeric_continuous"

    def test_socrata_number_sign_suffix(self, tmp_path: Path) -> None:
        values = [3_000_000 + (i % 200) * 113 for i in range(400)]
        detections = table(tmp_path, ["job__", "x"], [[v, i] for i, v in enumerate(values)])

        assert detections["job__"].type == "categorical_nominal"

    def test_a_number_named_by_its_head_word_is_a_code(self, tmp_path: Path) -> None:
        # "item_nbr": the quantity word is the head word, so the name says an item number
        values = [100_000 + (i % 90) * 7 for i in range(400)]
        detections = table(tmp_path, ["item_nbr", "x"], [[v, i] for i, v in enumerate(values)])

        assert detections["item_nbr"].type == "categorical_nominal"
        assert coded(detections["item_nbr"], "R-TYP-04")

    def test_a_measure_named_by_another_word_stays_a_measure(self, tmp_path: Path) -> None:
        # "YearsCode" counts years of coding: "years" says the measure, not "code"
        values = [str(i % 45) for i in range(400)]
        detections = table(tmp_path, ["YearsCode", "x"], [[v, i] for i, v in enumerate(values)])

        assert detections["YearsCode"].type in ("numeric_continuous", "numeric_discrete")

    def test_unix_times_are_no_codes(self, tmp_path: Path) -> None:
        values = [1_700_000_000 + i * 3_607 for i in range(400)]
        detections = table(
            tmp_path, ["last_modified_t", "x"], [[v, i] for i, v in enumerate(values)]
        )

        assert detections["last_modified_t"].type == "numeric_continuous"

    def test_numbered_stores_written_as_text_are_categories(self, tmp_path: Path) -> None:
        detections = table(tmp_path, ["store", "x"], [[f"{i % 45 + 1}", i] for i in range(400)])

        assert (detections["store"].type, detections["store"].role) == (
            "categorical_nominal",
            "entity",
        )


class TestHeadWord:
    @pytest.mark.parametrize(
        ("name", "head"),
        [
            ("patient_nbr", "nbr"),
            ("Code du département", "code"),
            ("Code de la catégorie socio-professionnelle", "code"),
            ("Lot Area", "area"),
            ("MS SubClass", "class"),
            ("Mother's occupation", "occupation"),
        ],
    )
    def test_head_word(self, name: str, head: str) -> None:
        assert head_word(name) == head


class TestDates:
    """R-TYP-06, R-TYP-07, year role (TYP-5, TYP-6, TYP-7)."""

    def test_two_date_formats(self, tmp_path: Path) -> None:
        dates = [
            f"{i % 12 + 1:02d}/{i % 28 + 1:02d}/2021" if i % 3 else f"20{i % 20:02d}-11-08"
            for i in range(300)
        ]
        detections = table(tmp_path, ["issuance_date", "n"], [[d, i] for i, d in enumerate(dates)])

        when = detections["issuance_date"]
        assert when.type == "datetime"
        assert coded(when, "R-TYP-06")
        assert len(when.parser.formats) == 2
        assert invalid_count(tmp_path, when) == 0

    def test_two_formats_that_would_lose_values_stay_text(self, tmp_path: Path) -> None:
        # 2 % of Excel serial numbers: a date reading would empty them (left to Clean, B-11)
        dates = [
            "44562" if i % 50 == 0 else (f"0{i % 9 + 1}/12/2021" if i % 2 else "2021-11-08")
            for i in range(300)
        ]
        detections = table(tmp_path, ["signup", "n"], [[d, i] for i, d in enumerate(dates)])

        assert detections["signup"].type != "datetime"

    def test_time_of_day_text(self, tmp_path: Path) -> None:
        times = [f"{i % 24:02d}.{(i * 7) % 60:02d}.00" for i in range(300)]
        detections = table(tmp_path, ["Time", "n"], [[t, i] for i, t in enumerate(times)])

        time = detections["Time"]
        assert time.type == "numeric_continuous"
        assert time.parser.kind == "time"
        assert time.role == "hour"
        assert coded(time, "R-TYP-07")

    def test_time_storage_is_read_as_hours(self, tmp_path: Path) -> None:
        connection = duckdb.connect()
        connection.execute(
            "CREATE TABLE t AS SELECT (TIME '00:00:00' + INTERVAL (range * 7) MINUTE) AS hrmn, "
            "range AS n FROM range(300)"
        )
        detection = next(d for d in detect_columns(connection, "t") if d.name == "hrmn")

        assert detection.type == "numeric_continuous"
        assert detection.parser.kind == "time"
        value = connection.execute(
            f"SELECT {typed_sql('hrmn', detection.parser)} FROM t WHERE n = 10"
        ).fetchone()
        assert value is not None and abs(value[0] - 70 / 60) < 1e-9

    def test_old_years_are_years(self, tmp_path: Path) -> None:
        detections = table(tmp_path, ["year", "co2"], [[1750 + i, i * 1.5] for i in range(275)])

        assert (detections["year"].type, detections["year"].role) == ("datetime", "year")


class TestRoles:
    """R-TYP-08 (TYP-8)."""

    def test_state_plane_coordinates_are_no_region(self, tmp_path: Path) -> None:
        values = [913_000 + i * 397 for i in range(400)]
        detections = table(
            tmp_path, ["x_coordinate_state_plane", "n"], [[v, i] for i, v in enumerate(values)]
        )

        assert detections["x_coordinate_state_plane"].role != "region"
        assert detections["x_coordinate_state_plane"].type == "numeric_continuous"

    def test_a_url_role_needs_urls(self, tmp_path: Path) -> None:
        values = ["NOT APPLICABLE", "ON-SITE", "OFF-SITE", "USE UNDER 300 CU.YD"]
        detections = table(tmp_path, ["site_fill", "n"], [[values[i % 4], i] for i in range(200)])

        assert detections["site_fill"].role != "url"

    def test_the_first_word_wins_a_tie(self) -> None:
        role = load().match_role("PercentSalaryHike")

        assert role is not None and role.id == "percentage"


class TestListsRangesCounts:
    """R-TYP-09, R-TYP-10, R-TYP-11 (TYP-9, TYP-10, KPI-8)."""

    def test_multiple_choice_lists(self, tmp_path: Path) -> None:
        choices = ["Python", "SQL", "Rust", "Go", "Java", "C", "R", "Bash", "JavaScript", "Kotlin"]
        values = [
            ";".join(choices[j] for j in range(10) if (i >> j) & 1) or "Python" for i in range(400)
        ]
        detections = table(
            tmp_path, ["LanguageHaveWorkedWith", "n"], [[v, i] for i, v in enumerate(values)]
        )

        languages = detections["LanguageHaveWorkedWith"]
        assert (languages.type, languages.role) == ("text", "choices")
        assert coded(languages, "R-TYP-09")

    def test_bracket_ranges_are_ordered(self, tmp_path: Path) -> None:
        ranges = [f"[{low}-{low + 10})" for low in range(0, 100, 10)]
        detections = table(tmp_path, ["age", "n"], [[ranges[(i * 7) % 10], i] for i in range(300)])

        age = detections["age"]
        assert age.type == "categorical_ordinal"
        assert age.order == ranges
        assert coded(age, "R-TYP-10")

    def test_worded_ranges_are_ordered(self, tmp_path: Path) -> None:
        ranges = ["Under 18 years old", "18-24 years old", "25-34 years old", "35-44 years old",
                  "65 years or older", "Prefer not to say"]  # fmt: skip
        detections = table(tmp_path, ["Age", "n"], [[ranges[(i * 5) % 6], i] for i in range(300)])

        assert detections["Age"].type == "categorical_ordinal"
        assert detections["Age"].order == ranges

    def test_numbered_labels_are_no_ranges(self, tmp_path: Path) -> None:
        labels = [f"Product {i}" for i in range(1, 9)]
        detections = table(tmp_path, ["product", "n"], [[labels[i % 8], i] for i in range(300)])

        assert detections["product"].type == "categorical_nominal"

    def test_a_count_of_zero_or_one_stays_a_count(self, tmp_path: Path) -> None:
        detections = table(
            tmp_path, ["number_of_cyclist_killed", "n"], [[int(i % 50 == 0), i] for i in range(400)]
        )

        assert detections["number_of_cyclist_killed"].type == "numeric_discrete"
        assert coded(detections["number_of_cyclist_killed"], "R-TYP-11")
