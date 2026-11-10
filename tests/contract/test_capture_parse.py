"""Fecha de captura desde el nombre del archivo (usada por ia-fake e IA-20)."""

from datetime import datetime

import pytest

from wwm_pipeline.capture_parse import parse_datetime_from_filename


@pytest.mark.parametrize(
    "name,expected",
    [
        ("CAM01_20260315_042210.mp4", datetime(2026, 3, 15, 4, 22, 10)),
        ("IMG_20260315-042210.jpg", datetime(2026, 3, 15, 4, 22, 10)),
        ("20260315042210.AVI", datetime(2026, 3, 15, 4, 22, 10)),
        ("2026-03-15 04-22-10.avi", datetime(2026, 3, 15, 4, 22, 10)),
        ("2026-03-15_04.22.10.MOV", datetime(2026, 3, 15, 4, 22, 10)),
        ("VID_2026_03_15_04_22_10.mp4", datetime(2026, 3, 15, 4, 22, 10)),
        (
            "WhatsApp Video 2026-05-30 at 9.49.20 AM.mp4",
            None,
        ),  # sin segundos fijos con 2 dígitos de hora: no se adivina
        ("IMG_0042.JPG", None),
        ("CAM01_20261340_250000.mp4", None),  # fecha imposible
        ("CAM01_19000101_000000.mp4", None),  # año anterior a 1990
    ],
)
def test_parse_datetime_from_filename(name, expected):
    assert parse_datetime_from_filename(name) == expected
