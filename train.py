import torch
import torch.nn.utils.rnn as U
from collections import defaultdict
from utils.train_util import prepare_mini, calc_metrics

def train_util(model, data, train=False, optimizer=None, batch_size=5, device='cuda'):
    if train:  model.train()
    else:      model.eval()

    nb = 0
    loss = 0
    metrics = {}
    metrics_tracker = defaultdict(lambda: torch.zeros((model.num_labels,), device=device))

    def update_metrics_tracker(preds, labels):
        match = preds * labels
        metrics_tracker['preds'] += torch.sum(preds, dim=0)
        metrics_tracker['labels'] += torch.sum(labels, dim=0)
        metrics_tracker['match'] += torch.sum(match, dim=0)

    for batch in prepare_mini(data, batch_size, device, train):
        if 'cuda' in device:
            torch.cuda.empty_cache()

        model_out = model(batch)
        if train == True:
            optimizer.zero_grad()
            model_out['loss'].backward()
            optimizer.step()

        update_metrics_tracker(model_out['doc_preds'], batch['doc_labels'])
        loss += model_out['loss'].item()
        nb += 1

    metrics['loss'] = loss / nb
    metrics.update(calc_metrics(metrics_tracker))

    return metrics


def train(model, train_data, dev_data, optimizer, lr_scheduler=None, epochs=100, batch_size=5, device='cuda'):
    curr_best = {'macro-F1': 0}
    best_model = model.state_dict()

    print('Training...\n')
    print("%5s || %8s | %8s || %8s | %8s %8s %8s" % ('EPOCH', 'Tr-LOSS', 'Tr-F1', 'Dv-LOSS', 'Dv-P', 'Dv-R', 'Dv-F1'))

    for epoch in range(epochs):
        train_metrics = train_util(model, train_data, train=True, optimizer=optimizer, batch_size=batch_size, device=device)
        dev_metrics = train_util(model, dev_data, batch_size=batch_size, device=device)
        if lr_scheduler:  lr_scheduler.step(dev_metrics['macro-F1'])
        print("%5d || %8.4f | %8.4f || %8.4f | %8.4f %8.4f %8.4f" % (epoch+1, train_metrics['loss'], train_metrics['macro-F1'], dev_metrics['loss'], dev_metrics['macro-P'], dev_metrics['macro-R'], dev_metrics['macro-F1']))

        if dev_metrics['macro-F1'] > curr_best['macro-F1']:
            curr_best = dev_metrics
            best_model = model.state_dict()

    print('\nTraining Completed\n')
    print("%5s || %8s | %8s || %8.4f | %8.4f %8.4f %8.4f" % ('Best', '-', '-', curr_best['loss'], curr_best['macro-P'], curr_best['macro-R'], curr_best['macro-F1']))

    return curr_best, best_model

