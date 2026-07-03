"""Tests for structured-output assertions.

These tests are pure Python -- no torch, no model loading. They validate
returns_valid_json(), matches_schema(), and has_keys().
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm_behave.core.assertions import assert_behavior


class TestReturnsValidJson:
    def test_plain_object_passes(self):
        assert_behavior('{"status": "ok", "count": 3}').returns_valid_json()

    def test_array_passes(self):
        assert_behavior("[1, 2, 3]").returns_valid_json()

    def test_fenced_json_passes(self):
        text = '```json\n{"status": "ok"}\n```'
        assert_behavior(text).returns_valid_json()

    def test_fenced_without_lang_passes(self):
        text = '```\n{"status": "ok"}\n```'
        assert_behavior(text).returns_valid_json()

    def test_invalid_json_fails(self):
        with pytest.raises(AssertionError, match="not valid JSON"):
            assert_behavior("this is not json").returns_valid_json()

    def test_trailing_comma_fails(self):
        with pytest.raises(AssertionError):
            assert_behavior('{"a": 1,}').returns_valid_json()


class _Order(BaseModel):
    order_id: str
    total: float
    shipped: bool


class TestMatchesSchema:
    def test_valid_payload_passes(self):
        text = '{"order_id": "A1", "total": 9.99, "shipped": true}'
        assert_behavior(text).matches_schema(_Order)

    def test_fenced_payload_passes(self):
        text = '```json\n{"order_id": "A1", "total": 9.99, "shipped": true}\n```'
        assert_behavior(text).matches_schema(_Order)

    def test_missing_field_fails(self):
        text = '{"order_id": "A1", "total": 9.99}'
        with pytest.raises(AssertionError, match="does not match schema"):
            assert_behavior(text).matches_schema(_Order)

    def test_wrong_type_fails(self):
        text = '{"order_id": "A1", "total": "free", "shipped": true}'
        with pytest.raises(AssertionError, match="does not match schema"):
            assert_behavior(text).matches_schema(_Order)

    def test_non_json_fails(self):
        with pytest.raises(AssertionError):
            assert_behavior("nope").matches_schema(_Order)


class TestHasKeys:
    def test_all_keys_present_passes(self):
        text = '{"name": "Sam", "email": "s@x.com", "age": 30}'
        assert_behavior(text).has_keys(["name", "email"])

    def test_missing_key_fails(self):
        text = '{"name": "Sam"}'
        with pytest.raises(AssertionError, match="missing"):
            assert_behavior(text).has_keys(["name", "email"])

    def test_non_object_fails(self):
        with pytest.raises(AssertionError, match="JSON object"):
            assert_behavior("[1, 2, 3]").has_keys(["name"])

    def test_invalid_json_fails(self):
        with pytest.raises(AssertionError):
            assert_behavior("not json").has_keys(["name"])


class TestChaining:
    def test_structured_asserts_chain(self):
        text = '{"name": "Sam", "email": "s@x.com"}'
        result = (
            assert_behavior(text)
            .returns_valid_json()
            .has_keys(["name", "email"])
        )
        assert len(result.results) == 2
        assert all(r.passed for r in result.results)
