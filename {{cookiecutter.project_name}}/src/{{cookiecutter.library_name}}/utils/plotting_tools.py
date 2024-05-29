"""Functions for plotting various aspect of the ML train/eval pipeline."""

from typing import Any, List, Dict

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_prediction_accuracy(accuracy_results: List) -> matplotlib.figure:
	"""Create a set of plots visualizing an output predictor's performances."""
	first_element = accuracy_results[0]
	n_features = len(list(first_element.keys()))

	# First plot: next-step output forecast
	fig, axes = plt.subplots(
		n_features, figsize=(10, 2 * n_features), gridspec_kw={"hspace": 0.5}
	)
	fig.suptitle("Output Predictor Next-Step Forecast")

	for i, k in enumerate(first_element.keys()):
		if n_features == 1:
			ax = axes
		else:
			ax = axes[i]

		ax.set_title(k)
		x = [v[k]["x"] for v in accuracy_results]
		yt = [v[k]["truth"][0, 0] for v in accuracy_results]
		yp = [v[k]["predi"][0, 0] for v in accuracy_results]

		ax.plot(x, yt, label="truth")
		ax.plot(x, yp, label="prediction")

		handles, lbs = ax.get_legend_handles_labels()
	fig.legend(handles, lbs, loc="center right")

	return fig


def plot_anomaly_latent_space(self, training_set, test_set) -> matplotlib.figure:
	"""Latent space visualization.

	We plot the Latent space visualization of the training
	and test dataset, using the PCA decomposition function
	fitted to the train data.

	The goal of this plot is to confirm that the test data
	maps out the same regions of the latent space as the
	training data

	"""
	fig, ax = plt.subplots(figsize=(10, 10))

	train_samples = []
	for i, d in enumerate(training_set):
		d = torch.unsqueeze(d, dim=0)
		latent_representation = self.model.encoder(d)
		array = latent_representation.cpu().detach().numpy()
		train_samples.append(array.flatten())

	train_samples = np.array(train_samples)
	pca = PCA(n_components=2)
	pca_model = pca.fit(train_samples)
	reduced_values_train = pca_model.transform(train_samples)

	# plot the training set first
	ax.scatter(
		reduced_values_train[:, 0],
		reduced_values_train[:, 1],
		label="training data",
		alpha=0.3,
	)

	test_samples = []
	for i, d in enumerate(test_set):
		d = torch.unsqueeze(d, dim=0)
		latent_representation = self.model.encoder(d)
		array = latent_representation.cpu().detach().numpy()
		test_samples.append(array.flatten())

	test_samples = np.array(test_samples)
	pca = PCA(n_components=2)
	reduced_values_test = pca_model.transform(test_samples)
	ax.scatter(
		reduced_values_test[:, 0],
		reduced_values_test[:, 1],
		label="test data",
		alpha=0.3,
	)
	ax.legend()
	return fig


def plot_summaries(summaries: Dict) -> List:
	"""Plot summaries of results across several trainings.

	This generates one plot per feature used across all
	trainings.
	"""
	summary_first = list(summaries.values())[0]

	metrics = ("mean_abs_error", "mean_sq_error", "rms_error")
	features = ("LIST YOUR FEATURES HERE.",)

	figlist = []
	for feature in features:
		fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
		ax.set_ylabel('Error magnitude')
		ax.set_title(f'performance of the training for feature: {feature}')
		width = 0.25
		multiplier = 0

		relevant_train_labels = []
		for training_label, v in summaries.items():
			if feature not in v.keys():
				continue
			relevant_train_labels.append(training_label)

		relevant_results: Dict={k: [] for k in relevant_train_labels}
		for training in relevant_train_labels:
			for metric in metrics:
				relevant_results[training].append(summaries[training][feature][metric])

		x = np.arange(len(metrics))
		for attribute, measurement in relevant_results.items():
			offset = width * multiplier
			rects = ax.bar(x + offset, measurement, width, label=attribute)
			ax.bar_label(rects, padding=3)
			multiplier += 1

		# Add some text for labels, title and custom x-axis tick labels, etc.

		ax.set_xticks(x + width, metrics)
		ax.legend(loc='upper left')
		figlist.append(fig)

	return figlist


