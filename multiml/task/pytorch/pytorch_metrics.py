import numpy as np

import torch
from torch import Tensor


def dummy(*args, **kwargs):
    return None


class EpochMetric:
    """Utility class to manage epoch metrics."""
    def __init__(self, metrics, enable, ml):
        self.metrics = metrics
        self.enable = enable
        self.ml = ml
        self.total = 0
        self.buffs = []
        self.preds = []

    def __call__(self, batch_result):
        result = {}
        if not self.enable:
            return result

        batch_size = batch_result['batch_size']
        self.total += batch_size

        for ii, metric in enumerate(self.metrics):
            if isinstance(metric, str):
                metric_fn = getattr(self, metric, dummy)
            else:
                metric_fn = metric
                metric = metric_fn.__name__

            metric_result = metric_fn(batch_result)

            if self.total == batch_size:  # first batch
                if isinstance(metric_result, list):
                    self.buffs.append([jmetric * batch_size for jmetric in metric_result])
                else:
                    self.buffs.append(metric_result * batch_size)
                result[metric] = metric_result

            else:
                if isinstance(metric_result, list):
                    for jj, jmetric in enumerate(metric_result):
                        self.buffs[ii][jj] += jmetric * batch_size
                    result[metric] = [jmetric / self.total for jmetric in self.buffs[ii]]

                else:
                    self.buffs[ii] += metric_result * batch_size
                    result[metric] = self.buffs[ii] / self.total

        return result

    # metrics
    def loss(self, batch_result):
        return batch_result['loss']['loss'].detach().item()

    def subloss(self, batch_result):
        result = []
        for index, subloss in enumerate(batch_result['subloss']):
            result.append(batch_result['loss']['subloss'].detach().item())
        return result

    def acc(self, batch_result):
        outputs = batch_result['outputs']
        labels = batch_result['labels']

        if isinstance(outputs, list):
            result = []
            for output, label in zip(outputs, labels):
                _, preds = torch.max(output, 1)
                corrects = torch.sum(preds == label.data)
                result.append(corrects.detach().item())
        else:
            _, preds = torch.max(outputs, 1)
            corrects = torch.sum(preds == labels.data)
            result = corrects.detach().item()
        return result / len(labels)

    def lr(self, batch_result):
        return [p['lr'] for p in self.ml.optimizer.param_groups]

    # ---

    def pred(self, batch_result):
        outputs = batch_result['pred']
        if isinstance(outputs, Tensor):
            self.preds.append(outputs.cpu().numpy())
        else:
            if self.preds:
                for index, output_obj in enumerate(outputs):
                    output_obj = output_obj.cpu().numpy()
                    self.preds[index].append(output_obj)
            else:
                for output_obj in outputs:
                    output_obj = output_obj.cpu().numpy()
                    self.preds.append([output_obj])

    def all_preds(self):
        if isinstance(self.preds[0], list):
            results = [np.concatenate(result, 0) for result in self.preds]
        else:
            results = np.concatenate(self.preds, 0)

        return results


def get_pbar_metric(epoch_result):
    result = {}
    for key, value in epoch_result.items():

        if isinstance(value, list):
            result[key] = [f'{v:.2e}' for v in value]
        elif isinstance(value, float):
            result[key] = f'{value:.2e}'
        else:
            result[key] = value
    return result
