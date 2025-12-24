#!/usr/bin/env python3
"""Plot accuracy vs components for strip layout experiments."""

import matplotlib.pyplot as plt
import numpy as np

# Data from experiments (components, accuracy estimate, time_seconds)
# Accuracy is approximate based on expert agreement
data = {
    'components': [9, 12, 16, 20, 24],
    'accuracy': [89, 83, 75, 75, 75],  # % agreement with expert
    'time': [17, 21, 25, 28, 31],  # seconds
    'status': ['stable', 'stable', 'stable', 'stable', 'stable']
}

# Additional data points
timeout_components = [27, 30]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Accuracy vs Components
ax1.plot(data['components'], data['accuracy'], 'bo-', linewidth=2, markersize=10, label='Accuracy')
ax1.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='80% threshold')
ax1.axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='70% threshold')

# Mark timeout zone
ax1.axvspan(25, 32, alpha=0.2, color='red', label='Timeout risk zone')

ax1.set_xlabel('Components per Image', fontsize=12)
ax1.set_ylabel('Classification Accuracy (%)', fontsize=12)
ax1.set_title('Strip Layout: Accuracy vs Components', fontsize=14)
ax1.set_xlim(8, 32)
ax1.set_ylim(50, 100)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower left')

# Add annotations
for i, (comp, acc) in enumerate(zip(data['components'], data['accuracy'])):
    ax1.annotate(f'{acc}%', (comp, acc), textcoords="offset points", 
                xytext=(0, 10), ha='center', fontsize=10)

# Plot 2: Time vs Components
ax2.plot(data['components'], data['time'], 'ro-', linewidth=2, markersize=10, label='Response time')

# Add timeout markers
ax2.scatter([27], [95], marker='x', s=200, c='red', linewidths=3, label='With retries', zorder=5)
ax2.scatter([30], [100], marker='X', s=200, c='darkred', linewidths=3, label='Timeout (failed)', zorder=5)

# Cloudflare timeout line
ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='~100s Cloudflare timeout')

ax2.set_xlabel('Components per Image', fontsize=12)
ax2.set_ylabel('Response Time (seconds)', fontsize=12)
ax2.set_title('Strip Layout: Response Time vs Components', fontsize=14)
ax2.set_xlim(8, 32)
ax2.set_ylim(0, 120)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')

# Add annotations for time
for i, (comp, t) in enumerate(zip(data['components'], data['time'])):
    ax2.annotate(f'{t}s', (comp, t), textcoords="offset points", 
                xytext=(0, 10), ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('experiments/grid_tests/accuracy_vs_components.png', dpi=150, bbox_inches='tight')
plt.savefig('experiments/grid_tests/accuracy_vs_components.pdf', bbox_inches='tight')
print("Saved: experiments/grid_tests/accuracy_vs_components.png")
print("Saved: experiments/grid_tests/accuracy_vs_components.pdf")
