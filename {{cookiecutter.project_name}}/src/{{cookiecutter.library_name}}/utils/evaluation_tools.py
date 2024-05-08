"""Set of functions used to evlauate performances of an ML model."""

import numpy as np
from dateutil.parser import parse
from typing import List, Dict
from torch.utils.data import DataLoader

from {{ cookiecutter.library_name }}.ml_tools.datasets import {{ cookiecutter.class_prefix }}Dataset
from {{ cookiecutter.library_name }}.ml_tools.models import {{ cookiecutter.class_prefix }}LSTM



def prediction_accuracy(model: {{ cookiecutter.class_prefix }}LSTM, dataset: {{ cookiecutter.class_prefix }}Dataset)-> List:
    """Evaluate the accuracy of the model's prediction,
    for a given dataset
    """
    model.cpu()

    accuracy_results = []

    dloader = DataLoader(dataset, batch_size=1)
    for y in dloader:
        current_pt = y["current_pt"]
        input_data = y["input"]
        truth_data = y["truth"]

        prediction = model(input_data)

        result = {}
        for k in truth_data.keys():
            truth = truth_data[k].detach().numpy()
            predi = prediction[k].detach().numpy()
            diff = (predi - truth)
            percent_diff = diff / (truth) * 100.0

            result[k] = {
                "x": parse(current_pt["time"][0][0]),
                "truth": truth,
                "predi": predi,

                # difference
                "diff": diff,
                "diff_mean": diff.mean(),

                # absolute difference
                "abs_diff": np.abs(diff),
                "abs_diff_tot": np.abs(diff).sum(),
                "mae": np.abs(diff).mean(),
                "mse": (diff**2.0).mean(),
                "rms": np.sqrt((diff**2.0).mean()),

                # percent difference
                "percent_diff": percent_diff,
                "percent_diff_mean": percent_diff.mean(),
                "percent_abs_diff": np.abs(percent_diff),
                "percent_abs_diff_mean": np.abs(percent_diff).mean(),
                }
        accuracy_results.append(result)

    return accuracy_results


def summarize_training_accuracy(accuracy_results: List) -> Dict:
	"""Compute dataset-wide performance metrics to summarize model accuracy.

	All values are computed across all predicted future time steps.
	"""
	first_element = accuracy_results[0]
	summary = {}
	for k in first_element.keys():
		total_abs_diff = np.sum([v[k]["abs_diff_tot"] for v in accuracy_results])
		mean_abs_error = np.mean([v[k]["mae"] for v in accuracy_results])
		mean_sq_error = np.mean([v[k]["mse"] for v in accuracy_results])
		rms_error = np.sqrt(np.mean(np.concatenate([np.array(v[k]["diff"])**2. for v in accuracy_results])))
		mean_percent = np.mean([v[k]["percent_diff"] for v in accuracy_results])

		summary[k] = {
			"total_abs_diff": total_abs_diff,
			"mean_abs_error": mean_abs_error,
			"mean_sq_error": mean_sq_error,
			"rms_error":rms_error,
			"mean_percent": mean_percent
		}

	return summary
