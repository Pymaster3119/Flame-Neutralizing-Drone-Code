from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt


EPOCH_PATTERN = re.compile(r"^Epoch\s+(\d+)/(\d+)")
TRAIN_LOSS_PATTERN = re.compile(r"^Training Loss:\s*([0-9]*\.?[0-9]+)")
VAL_LOSS_PATTERN = re.compile(r"^Validation Loss:\s*([0-9]*\.?[0-9]+)")


def parse_losses(log_path):
	epochs: list[int] = []
	train_losses: list[float] = []
	val_losses: list[float] = []

	current_epoch = None

	for raw_line in log_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()

		epoch_match = EPOCH_PATTERN.match(line)
		if epoch_match:
			current_epoch = int(epoch_match.group(1))
			continue

		train_match = TRAIN_LOSS_PATTERN.match(line)
		if train_match and current_epoch is not None:
			epochs.append(current_epoch)
			train_losses.append(float(train_match.group(1)))
			continue

		val_match = VAL_LOSS_PATTERN.match(line)
		if val_match:
			val_losses.append(float(val_match.group(1)))

	return epochs, train_losses, val_losses


def plot_losses(epochs, train_losses, val_losses, output_path):
	plt.figure(figsize=(8, 5))
	plt.plot(epochs, train_losses, marker="o", label="Training Loss")
	plt.plot(epochs, val_losses, marker="s", label="Validation Loss")
	plt.title("Training vs Validation Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.grid(alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()

if __name__ == "__main__":
	epochs, train_losses, val_losses = parse_losses(Path('input.txt'))
	plot_losses(epochs, train_losses, val_losses, 'loss_curve.png')
