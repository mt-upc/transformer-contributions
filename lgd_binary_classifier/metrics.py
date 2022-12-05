import torch
from sklearn.metrics import accuracy_score, f1_score

class MetricsLogger:
    base_buffer = {
        'loss': [],
        'accuracy': [],
        'f1': [],
    }
    def __init__(self, buffer_names, loss_fn):
        self.buffers = {
            b: self.base_buffer.copy() for b in buffer_names
        }
        self.loss_fn = loss_fn

    def reset(self, buffer_name):
        self.buffers[buffer_name] = {
            k: [] for k, v in self.buffers[buffer_name].items()
        }

    def add(self, net_out, target):
        net_out = net_out.detach().cpu()
        target = target.cpu()
        pred = torch.sigmoid(net_out).round()
        for buf in self.buffers.values():
            buf['loss'].append(self.loss_fn(net_out, target.float()).item())
            buf['accuracy'].append(accuracy_score(target, pred))
            buf['f1'].append(f1_score(target, pred, zero_division=0))

    def get_avg(self, buffer_name):
        return {
            k: sum(v) / len(v) for k, v in self.buffers[buffer_name].items()
        }
