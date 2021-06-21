import torch


class BatchMetric:
    @staticmethod
    def dummy(outputs, labels, loss):
        return None

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
    def __init__(self, true_var_names, ml):
        self.ml = ml
        self.total = 0
        self.epoch_loss = 0.0
        self.epoch_corrects = 0.0

        if isinstance(true_var_names, list):
            num_subtasks = len(true_var_names)
            self.epoch_subloss = [0.0] * num_subtasks
            self.epoch_corrects = [0] * num_subtasks

    def dummy(self, batch_result):
        return None

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


def get_pbar_metric(epoch_result):
    result = {}
    for key, value in epoch_result.items():
        if isinstance(value, list):
            result[key] = [f'{v:.2e}' for v in value]
        else:
            result[key] = value
    return result
