from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .data import GROUPS, fmt

for _font in sorted((Path(__file__).parent / 'fonts').glob('*.ttf')):
    font_manager.fontManager.addfont(str(_font))
logging.getLogger('fontTools').setLevel(logging.ERROR)
plt.rcParams.update({'font.family': 'Carlito', 'pdf.fonttype': 42,
                     'svg.fonttype': 'none'})

LEVEL_COLORS = ('#94b8cc', '#6597b2', '#3a7898', '#156082')
COMPONENT_COLORS = {'classification': '#94b8cc', 'extraction': '#6597b2', 'matching': '#156082'}
TEXT, MUTED, GRID, HIGHLIGHT = '#1f1f1e', '#6b6a63', '#e3e2dc', '#cfe3f2'
BASELINE_GROUP = 'NLP baselines'
WIDTH = 370.6 / 72


def _best_per_group(rows):
    best = {}
    for row in rows:
        best[row['group']] = max(best.get(row['group'], -1), round(row['overall'], 9))
    return {i for i, row in enumerate(rows) if round(row['overall'], 9) == best[row['group']]}


def _axes_style(ax, grid_axis):
    ax.grid(axis=grid_axis, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=MUTED)


def difficulty_figure(rows, xlabel, path):
    groups = [g for g in (*GROUPS, BASELINE_GROUP) if any(r['group'] == g for r in rows)]
    layout, y = [], 0.0
    for group in groups:
        layout.append(('header', group, y))
        y += 0.75
        for index, row in enumerate(rows):
            if row['group'] == group:
                layout.append(('row', index, y))
                y += 1.0
    best = _best_per_group(rows)
    fig, ax = plt.subplots(figsize=(WIDTH, 410.4 / 72))
    ticks, labels = [], []
    for kind, item, pos in layout:
        if kind == 'header':
            ax.text(-1.2, pos, item, ha='right', va='center', style='italic', fontsize=8, color=MUTED)
            continue
        row = rows[item]
        left, outside = 0.0, []
        for level, (value, color) in enumerate(zip(row['levels'].values(), LEVEL_COLORS)):
            width = value / 4
            ax.barh(pos, width, left=left, height=0.94, color=color, lw=0, zorder=2)
            if width >= 6:
                ax.text(left + width / 2, pos, fmt(value), ha='center', va='center', fontsize=8,
                        color='white' if level >= 2 else TEXT, zorder=3)
            else:
                outside.append(value)
            left += width
        for number, value in enumerate(outside):
            ax.text(left + 1.5 + 8.8 * number, pos, fmt(value), ha='left', va='center', fontsize=8, color=TEXT, zorder=3)
        bbox = dict(boxstyle='square,pad=0.3', facecolor=HIGHLIGHT, edgecolor='none') if item in best else None
        ax.text(107.5, pos, fmt(row['overall']), ha='center', va='center', fontsize=8.5, color=TEXT,
                fontweight='bold' if item in best else 'normal', bbox=bbox, clip_on=False)
        ticks.append(pos)
        labels.append(row['name'])
    ax.text(107.5, 0, 'Overall', ha='center', va='center', fontsize=8.5, fontweight='bold', color=TEXT, clip_on=False)
    ax.set_yticks(ticks, labels, fontsize=9, color=TEXT)
    ax.tick_params(axis='y', length=0, labelcolor=TEXT)
    ax.set_ylim(y + 0.4, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 20))
    _axes_style(ax, 'x')
    ax.tick_params(axis='y', labelcolor=TEXT)
    ax.tick_params(axis='x', labelsize=8.5)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT)
    handles = [Patch(facecolor=c, label=f'Difficulty {i}') for i, c in enumerate(LEVEL_COLORS, 1)]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8, handlelength=1.6,
               handleheight=1.0, bbox_to_anchor=(0.56, 0.0))
    fig.subplots_adjust(left=0.29, right=0.903, top=0.985, bottom=0.135)
    fig.savefig(path)
    plt.close(fig)


def runtime_figure(rows, path, limit=10.0, broken_height=12.8):
    fig, ax = plt.subplots(figsize=(WIDTH, 310.7 / 72))
    components = ('classification', 'extraction', 'matching')
    for x, row in enumerate(rows):
        broken = row['total'] > broken_height
        scale = (broken_height - row['matching']) / (row['classification'] + row['extraction']) if broken else 1.0
        bottom = 0.0
        for component in components:
            value = row[component]
            height = value if component == 'matching' else value * scale
            ax.bar(x, height, bottom=bottom, width=0.78, color=COMPONENT_COLORS[component], lw=0, zorder=2)
            if height >= 0.28:
                ax.text(x, bottom + height / 2, fmt(value, 2), ha='center', va='center', fontsize=7, fontweight='bold',
                        color=TEXT if component == 'classification' else 'white', zorder=3)
            if broken and component != 'matching':
                middle = bottom + height * 0.62
                ax.plot([x - 0.45, x + 0.45], [middle - 0.12, middle + 0.12], color='white', lw=2.5, zorder=4,
                        solid_capstyle='butt')
            bottom += height
        ax.text(x, bottom + 0.12, fmt(row['total'], 2), ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color=TEXT)
    ax.set_xticks(range(len(rows)), [r['name'] for r in rows], rotation=40, ha='right', rotation_mode='anchor', fontsize=8)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, broken_height + 0.9)
    ax.set_yticks(range(0, int(limit) + 1, 2))
    _axes_style(ax, 'y')
    ax.tick_params(axis='y', labelsize=8.5, length=0, labelcolor=TEXT)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('Runtime per report (s)', fontsize=9, color=TEXT)
    handles = [Patch(facecolor=COMPONENT_COLORS[c], label=c.capitalize()) for c in reversed(components)]
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=9, handlelength=1.6, handleheight=1.0,
              borderaxespad=0.0)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.985, bottom=0.215)
    fig.savefig(path)
    plt.close(fig)
