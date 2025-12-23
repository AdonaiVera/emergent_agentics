#!/usr/bin/env python3
import json
import os
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_json(file_path: str) -> Dict:
	try:
		with open(file_path, 'r', encoding='utf-8') as f:
			return json.load(f)
	except Exception as e:
		print(f"Error loading {file_path}: {e}")
		return {}


def extract_acceptance_history(data: Dict) -> List[Dict]:
	"""Return a flat list of acceptance/rejection events.
	Supports both dict-of-lists and list formats under
	acceptance_rejection.interaction_history."""
	df_like = data.get('acceptance_rejection', {})
	ih = df_like.get('interaction_history', {}) if isinstance(df_like, dict) else {}
	# Normalize to flat list of events
	if isinstance(ih, dict):
		events: List[Dict] = []
		for _, lst in ih.items():
			if isinstance(lst, list):
				for ev in lst:
					if isinstance(ev, dict):
						events.append(ev)
		return events
	elif isinstance(ih, list):
		return [ev for ev in ih if isinstance(ev, dict)]
	else:
		return []


def extract_interaction_counts(data: Dict) -> List[Dict]:
	# Expected under top-level key 'interaction_counts'
	df_like = data.get('interaction_counts', [])
	# If already a list of row dicts
	if isinstance(df_like, list):
		return [row for row in df_like if isinstance(row, dict)]
	# If a dict-of-dicts mapping source->targets, normalize to a single row
	if isinstance(df_like, dict):
		return [df_like]
	return []


def build_agents_from_acceptance(histories: List[List[Dict]]) -> List[str]:
	agents = set()
	for events in histories:
		if not isinstance(events, list):
			continue
		for event in events:
			if not isinstance(event, dict):
				continue
			initiator = event.get('initiator')
			target = event.get('target')
			if initiator:
				agents.add(initiator)
			if target:
				agents.add(target)
	return sorted(a for a in agents if a)


def build_agents_from_counts(all_counts_rows: List[List[Dict]]) -> List[str]:
	agents = set()
	for rows in all_counts_rows:
		for row in rows:
			for src, targets in row.items():
				if not isinstance(targets, dict):
					continue
				agents.add(src)
				agents.update(targets.keys())
	return sorted(a for a in agents if a)


def average_acceptance_ratio(json_files: List[str], save_path: str):
	# Load histories
	histories = [extract_acceptance_history(load_json(p)) for p in json_files]
	# Determine agents universe
	agents = build_agents_from_acceptance(histories)
	if not agents:
		print("No agents/interactions found in acceptance histories")
		return
	# Aggregate accepted and total counts across files
	accepted_sum = pd.DataFrame(0.0, index=agents, columns=agents)
	counts_sum = pd.DataFrame(0.0, index=agents, columns=agents)
	for events in histories:
		for event in events:
			initiator = event.get('initiator')
			target = event.get('target')
			if initiator not in accepted_sum.index or target not in accepted_sum.columns:
				continue
			counts_sum.loc[initiator, target] += 1.0
			if event.get('accepted'):
				accepted_sum.loc[initiator, target] += 1.0
	# Compute ratio
	with pd.option_context('mode.use_inf_as_na', True):
		ratio = accepted_sum.divide(counts_sum)
	mask = (counts_sum == 0)
	# Ensure diagonal is visible: set diagonal not masked and ratio to 0.0
	for i in range(len(agents)):
		mask.iloc[i, i] = False
		ratio.iloc[i, i] = 0.0
	# Plot heatmap (do not show)
	cmap = LinearSegmentedColormap.from_list("acceptance_gradient", ["red", "#d4af37", "green"])
	fig_w = 1.6*len(agents)+3
	fig_h = 1.6*len(agents)+1
	plt.figure(figsize=(fig_w, fig_h))
	ax = sns.heatmap(
		ratio,
		annot=True,
		fmt=".2f",
		cmap=cmap,
		vmin=0,
		vmax=1,
		mask=mask,
		linewidths=0.6,
		linecolor='gray',
		cbar_kws={"label": "Acceptance Ratio"},
		square=True,
		annot_kws={"color": "white", "fontsize": 12, "fontweight": "bold"}
	)
	# Make diagonal cells white and label with 0
	for i in range(len(agents)):
		# White overlay patch on diagonal cell
		ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='white', ec='gray', lw=0.6))
		# Draw a black '0' centered on the diagonal
		ax.text(i + 0.5, i + 0.5, '0', ha='center', va='center', fontsize=12, fontweight='bold', color='black')
	# Gray overlay for masked cells
	for y in range(ratio.shape[0]):
		for x in range(ratio.shape[1]):
			if mask.iloc[y, x]:
				ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, color='#e0e0e0', lw=0))
	plt.title("Average Acceptance Ratio Matrix (across files)")
	plt.ylabel("From (Initiator)")
	plt.xlabel("To (Target)")
	plt.tight_layout()
	# Save PNG and SVG
	base, ext = os.path.splitext(save_path)
	png_path = base + ".png"
	svg_path = base + ".svg"
	plt.savefig(png_path, dpi=300, bbox_inches='tight')
	plt.savefig(svg_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Saved: {png_path}\nSaved: {svg_path}")


def average_interaction_counts(json_files: List[str], save_path: str):
	# Load rows for each file
	all_counts_rows = [extract_interaction_counts(load_json(p)) for p in json_files]
	# Determine agents universe
	agents = build_agents_from_counts(all_counts_rows)
	if not agents:
		print("No agents/interactions found in interaction count data")
		return
	# For averaging, construct per-file matrix and average across files
	mat_sum = pd.DataFrame(0.0, index=agents, columns=agents)
	n_files = 0
	for rows in all_counts_rows:
		if not rows:
			continue
		n_files += 1
		mat = pd.DataFrame(0.0, index=agents, columns=agents)
		for row in rows:
			for src, targets in row.items():
				if not isinstance(targets, dict) or src not in agents:
					continue
				for tgt, count in targets.items():
					if tgt in agents and isinstance(count, (int, float)):
						mat.loc[src, tgt] += float(count)
		mat_sum += mat
	if n_files == 0:
		print("No valid interaction count rows found across files")
		return
	mat_avg = mat_sum / float(n_files)
	# Round for display
	mat_disp = mat_avg.round(0).astype(int)
	fig_w = 1.6*len(agents)+3
	fig_h = 1.6*len(agents)+1
	plt.figure(figsize=(fig_w, fig_h))
	sns.heatmap(
		mat_disp,
		annot=True,
		fmt="d",
		cmap="Blues",
		cbar_kws={"label": "Avg # Interactions"},
		linewidths=0.6,
		linecolor='gray',
		square=True,
		annot_kws={"color": "black", "fontsize": 12, "fontweight": "bold"}
	)
	plt.title("Average Interaction Count Matrix (across files)")
	plt.ylabel("From (Agent 1)")
	plt.xlabel("To (Agent 2)")
	plt.tight_layout()
	# Save PNG and SVG
	base, ext = os.path.splitext(save_path)
	png_path = base + ".png"
	svg_path = base + ".svg"
	plt.savefig(png_path, dpi=300, bbox_inches='tight')
	plt.savefig(svg_path, dpi=300, bbox_inches='tight')
	plt.close()
	print(f"Saved: {png_path}\nSaved: {svg_path}")


def discover_json_files(storage_dir: str, prefix: str, relative_json: str) -> List[str]:
	json_files: List[str] = []
	if not os.path.isdir(storage_dir):
		print(f"Storage directory not found: {storage_dir}")
		return json_files
	for name in os.listdir(storage_dir):
		full_path = os.path.join(storage_dir, name)
		if not os.path.isdir(full_path):
			continue
		if not name.startswith(prefix):
			continue
		target = os.path.join(full_path, relative_json)
		if os.path.isfile(target):
			json_files.append(target)
		else:
			print(f"Warning: JSON not found for {name}: {target}")
	return json_files


def discover_jsons_recursively(search_dir: str) -> List[str]:
	files: List[str] = []
	if not os.path.isdir(search_dir):
		print(f"Search directory not found: {search_dir}")
		return files
	for root, _, filenames in os.walk(search_dir):
		for fn in filenames:
			if fn.lower().endswith('.json'):
				files.append(os.path.join(root, fn))
	return files


def is_supported_format(data: Dict) -> bool:
	"""A file is supported if it has either acceptance history events or interaction_counts rows."""
	return bool(extract_acceptance_history(data)) or bool(extract_interaction_counts(data))


def main():
	parser = argparse.ArgumentParser(description='Average acceptance ratio and interaction count matrices across multiple JSON files')
	parser.add_argument('--json_files', nargs='+', help='List of JSON file paths to average')
	parser.add_argument('--search_dir', type=str, default='/home/ado/Documents/emergent_agentics/visualizations', help='Recursively search this directory for .json files')
	parser.add_argument('--auto_discover', action='store_true', help='Auto-discover JSON files under a storage directory')
	parser.add_argument('--storage_dir', type=str, default='/home/ado/Documents/emergent_agentics/environment/frontend_server/storage', help='Storage directory root for auto-discovery')
	parser.add_argument('--prefix', type=str, default='simulation_crosmodality_', help='Folder name prefix for auto-discovery')
	parser.add_argument('--relative_json', type=str, default='metrics.json', help='Relative JSON path inside each simulation folder')
	parser.add_argument('--save_dir', type=str, default='plots_avg', help='Directory to save averaged plots')
	args = parser.parse_args()
	# Determine JSON files (priority: search_dir -> json_files -> auto_discover)
	files: List[str] = []
	if args.search_dir:
		files = discover_jsons_recursively(args.search_dir)
		print(f"Recursively discovered {len(files)} JSON files under {args.search_dir}")
	elif args.json_files:
		files = args.json_files
	elif args.auto_discover:
		files = discover_json_files(args.storage_dir, args.prefix, args.relative_json)
		print(f"Discovered {len(files)} JSON files under {args.storage_dir} with prefix '{args.prefix}' and relative '{args.relative_json}'")
	else:
		print('Error: Provide --search_dir or --json_files or use --auto_discover with storage arguments')
		return
	if not files:
		print('No JSON files found to average.')
		return
	# Filter by format only (no step-count filtering)
	included: List[str] = []
	skipped: List[str] = []
	for fp in files:
		data = load_json(fp)
		if is_supported_format(data):
			included.append(fp)
		else:
			skipped.append(fp)
	print(f"Including {len(included)} files (supported format). Skipped {len(skipped)} files (unsupported format).")
	if skipped:
		for fp in skipped[:25]:
			print(f"  - Unsupported: {fp}")
		if len(skipped) > 25:
			print(f"  ... and {len(skipped)-25} more")
	if not included:
		print('No files with supported format; exiting.')
		return
	os.makedirs(args.save_dir, exist_ok=True)
	acc_out = os.path.join(args.save_dir, 'acceptance_ratio_matrix_avg.png')
	int_out = os.path.join(args.save_dir, 'interaction_count_matrix_avg.png')
	average_acceptance_ratio(included, acc_out)
	average_interaction_counts(included, int_out)
	print('Done.')

if __name__ == '__main__':
	main()
