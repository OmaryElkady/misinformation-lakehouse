"""
tests/unit/test_processing.py

Unit tests for pure logic in the Bronze→Silver and Silver→Gold processing jobs.
No Spark, no network — only the standalone helper functions are exercised here.
"""

import pytest

from src.processing.bronze_to_silver import normalize_label
from src.processing.silver_to_gold import get_source_credibility, safe_divide


class TestNormalizeLabel:
    @pytest.mark.unit
    def test_true_is_credible(self):
        assert normalize_label("true") == 0

    @pytest.mark.unit
    def test_mostly_true_is_credible(self):
        assert normalize_label("mostly-true") == 0

    @pytest.mark.unit
    def test_half_true_is_credible(self):
        assert normalize_label("half-true") == 0

    @pytest.mark.unit
    def test_real_is_credible(self):
        assert normalize_label("real") == 0

    @pytest.mark.unit
    def test_false_is_misinfo(self):
        assert normalize_label("false") == 1

    @pytest.mark.unit
    def test_pants_fire_is_misinfo(self):
        assert normalize_label("pants-fire") == 1

    @pytest.mark.unit
    def test_barely_true_is_misinfo(self):
        assert normalize_label("barely-true") == 1

    @pytest.mark.unit
    def test_fake_is_misinfo(self):
        assert normalize_label("fake") == 1

    @pytest.mark.unit
    def test_unknown_string_is_unknown(self):
        assert normalize_label("unknown") == -1

    @pytest.mark.unit
    def test_none_is_unknown(self):
        assert normalize_label(None) == -1

    @pytest.mark.unit
    def test_empty_string_is_unknown(self):
        assert normalize_label("") == -1

    @pytest.mark.unit
    def test_unlabeled_bluesky_is_unknown(self):
        assert normalize_label("unlabeled") == -1


class TestGetSourceCredibility:
    @pytest.mark.unit
    def test_liar(self):
        assert get_source_credibility("liar") == 0.7

    @pytest.mark.unit
    def test_fakenewsnet(self):
        assert get_source_credibility("fakenewsnet") == 0.6

    @pytest.mark.unit
    def test_bluesky(self):
        assert get_source_credibility("bluesky") == 0.3

    @pytest.mark.unit
    def test_unknown_source_returns_default(self):
        assert get_source_credibility("other") == 0.5

    @pytest.mark.unit
    def test_empty_string_returns_default(self):
        assert get_source_credibility("") == 0.5


class TestSafeDivide:
    @pytest.mark.unit
    def test_normal_division(self):
        assert safe_divide(10.0, 4.0) == 2.5

    @pytest.mark.unit
    def test_zero_denominator_returns_zero(self):
        assert safe_divide(5.0, 0.0) == 0.0

    @pytest.mark.unit
    def test_zero_numerator(self):
        assert safe_divide(0.0, 5.0) == 0.0

    @pytest.mark.unit
    def test_both_zero_returns_zero(self):
        assert safe_divide(0.0, 0.0) == 0.0
