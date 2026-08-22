# -*- coding: utf-8 -*-
"""Helpers for daily-vs-5min output dir remapping + comparison (Phase C)."""
import os


def output_dir_suffix(period):
    """'daily' -> '';  '5m' -> '_5m';  etc."""
    return '' if period == 'daily' else f'_{period}'


def remap_output_path(base_path, period):
    """data/projection/movement_xxx.csv → data/projection_5min/movement_xxx.csv (if period='5m')."""
    if period == 'daily':
        return base_path
    parent, fname = os.path.split(base_path)
    return os.path.join(parent.rstrip('/').rstrip('\\') + output_dir_suffix(period), fname)


def output_subdir_for_period(base, period):
    """backtrace/outputs/... → backtrace/outputs/..._5m (if period='5m')."""
    if period == 'daily':
        return base
    return base + output_dir_suffix(period)
