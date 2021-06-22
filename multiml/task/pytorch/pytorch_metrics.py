import numpy as np

import torch
from torch import Tensor


class BatchMetric:
    """ Utility class to manage batch metrics.
    """
    @staticmethod
    def loss(outputs, labels, loss):
        return loss['loss'].item()

    @staticmethod
    def subloss(outputs, labels, loss):
        return [l.item() for l in loss['subloss']]

    @staticmethod
    def acc(outputs, labels, loss):
        if isinstance(outputs, list):
            result = []
            for output, label in zip(outputs, labels):
                _, preds = torch.max(output, 1)
                corrects = torch.sum(preds == label.data)
                result.append(corrects.item())
        else:
            _, preds = torch.max(outputs, 1)
            corrects = torch.sum(preds == labels.data)
            result = corrects.item()
        return result


class EpochMetric:
    """ Utility class to manage epoch metrics.
    """
    def __init__(self, true_var_names, ml):
        self.ml = ml
        self.total = 0
        self.preds = []
        self.epoch_loss = 0.0
        self.epoch_corrects = 0.0

        if isinstance(true_var_names, list):
            num_subtasks = len(true_var_names)
            self.epoch_subloss = [0.0] * num_subtasks
            self.epoch_corrects = [0] * num_subtasks

    def loss(self, batch_result):
        self.epoch_loss += batch_result['loss'] * batch_result['batch_size']
        running_loss = self.epoch_loss / self.total
        return running_loss

    def subloss(self, batch_result):
        result = []
        for index, subloss in enumerate(batch_result['subloss']):
            self.epoch_subloss[index] += subloss * batch_result['batch_size']
            running_subloss = self.epoch_subloss[index] / self.total
            result.append(running_subloss)
        return result

    def acc(self, batch_result):
        if isinstance(batch_result['acc'], list):
            results = []
            for index, acc in enumerate(batch_result['acc']):
                self.epoch_corrects[index] += acc
                accuracy = self.epoch_corrects[index] / self.total
                results.append(accuracy)
            return results

        else:
            self.epoch_corrects += batch_result['acc']
            accuracy = self.epoch_corrects / self.total
            return accuracy

    def lr(self, batch_result):
        return [p['lr'] for p in self.ml.optimizer.param_groups]

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


def dummy(*args, **kwargs):
    return None


def get_pbar_metric(epoch_result):
    result = {}
    for key, value in epoch_result.items():
        if isinstance(value, list):
            result[key] = [f'{v:.2e}' for v in value]
        else:
            result[key] = f'{value:.2e}'
    return result
