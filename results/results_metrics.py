import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict

"""
This script was developed using Claude purely for visualization purposes, to create a plot summarizing the results from the CSV file.
It reads the CSV, computes min, max, and average for each metric by category and augmentation, and then creates a grid of bar plots with error bars representing the range and average values.
The final plot is saved as 'augmentation_plots.png'.
"""


CSV_PATH = 'results.csv'
OUTPUT_PATH = 'augmentation_plots.png'

data = defaultdict(list)
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row['category'].lower(), row['augmentation'].lower())
        data[key].append({
            'iou':  float(row['mean_iou']),
            'biou': float(row['mean_boundary_iou']),
            'diff': int(row['mask_count_diff']),
        })

stats = {}
for (cat, aug), rows in data.items():
    stats[(cat, aug)] = {
        'iou':  (min(r['iou']  for r in rows), max(r['iou']  for r in rows), sum(r['iou']  for r in rows) / len(rows)),
        'biou': (min(r['biou'] for r in rows), max(r['biou'] for r in rows), sum(r['biou'] for r in rows) / len(rows)),
        'diff': (min(r['diff'] for r in rows), max(r['diff'] for r in rows), sum(r['diff'] for r in rows) / len(rows)),
    }

AUGS          = ['gaussian', 'motion', 'compression', 'all']
AUG_LABELS    = {'gaussian': 'Gaussian', 'motion': 'Motion blur', 'compression': 'Compression', 'all': 'All augmentations'}
CATS          = ['dog', 'mobile_phone', 'train']
CAT_LABELS    = {'dog': 'Dog', 'mobile_phone': 'Mobile phone', 'train': 'Train'}
METRICS       = ['iou', 'biou', 'diff']
METRIC_LABELS = {'iou': 'IoU', 'biou': 'Boundary IoU', 'diff': 'Mask diff'}

COLORS = {'dog': '#7F77DD', 'mobile_phone': '#1D9E75', 'train': '#D85A30'}
EDGE   = {'dog': '#534AB7', 'mobile_phone': '#0F6E56', 'train': '#993C1D'}

fig, axes = plt.subplots(4, 3, figsize=(13, 14))
fig.patch.set_facecolor('#FAFAFA')

x = np.arange(len(CATS))
width = 0.55

for row_i, aug in enumerate(AUGS):
    for col_i, metric in enumerate(METRICS):
        ax = axes[row_i, col_i]
        ax.set_facecolor('white')

        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#CCCCCC')
        ax.tick_params(colors='#666666', labelsize=9)
        ax.yaxis.grid(True, color='#EEEEEE', linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        for xi, cat in enumerate(CATS):
            mn, mx, avg = stats[(cat, aug)][metric]
            ax.bar(xi, mx - mn, bottom=mn, width=width,
                   color=COLORS[cat], edgecolor=EDGE[cat], linewidth=0.8,
                   zorder=3, alpha=0.85)
            ax.plot(xi, avg, marker='x', markersize=8, markeredgewidth=2,
                    color=EDGE[cat], zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels([CAT_LABELS[c] for c in CATS], fontsize=9)

        if col_i == 0:
            ax.set_ylabel(AUG_LABELS[aug], fontsize=10, fontweight='bold', color='#333333', labelpad=8)
        if row_i == 0:
            ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight='bold', color='#333333', pad=8)

legend_patches = [mpatches.Patch(color=COLORS[c], label=CAT_LABELS[c]) for c in CATS]
legend_line = plt.Line2D([0], [0], marker='x', color='#555555', linestyle='None',
                         markersize=8, markeredgewidth=2, label='Average')
fig.legend(handles=legend_patches + [legend_line],
           loc='lower center', ncol=4, fontsize=10,
           frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle('IoU, Boundary IoU and Mask diff by augmentation and category',
             fontsize=13, fontweight='bold', color='#222222', y=0.995)

plt.tight_layout(rect=[0, 0.055, 1, 1])
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
print(f"Saved to {OUTPUT_PATH}")