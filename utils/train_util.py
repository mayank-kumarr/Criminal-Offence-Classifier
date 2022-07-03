import torch
import torch.nn.utils.rnn as U
from collections import defaultdict


def prepare_offence(data, device='cuda'):
    batch = defaultdict(list)

    for i in range(len(data)):
        ns = len(data[i]['text'])
        null_sents = 0
        for j in range(ns):
            sent = data[i]['text'][j]
            if len(sent) == 0:
                null_sents += 1
                continue

            batch['offence'].append(torch.tensor(sent, dtype=torch.long, device=device))
            batch['sent_lens'].append(len(sent))

        batch['doc_lens'].append(ns-null_sents)

    batch['offence'] = U.pad_sequence(batch['offence'], batch_first=True)
    batch['sent_lens'] = torch.tensor(batch['sent_lens'], dtype=torch.long, device=device)
    batch['doc_lens'] = torch.tensor(batch['doc_lens'], dtype=torch.long, device=device)

    return batch


def prepare_mini(data, batch_size=5, device='cuda', shuffle=False):
    if shuffle:
        data_idx = torch.randperm(len(data))
    else:
        data_idx = torch.arange(len(data))

    bsl = False
    bdl = False
    if 'sent_labels' in data[0]:  bsl = True
    if 'doc_labels'  in data[0]:  bdl = True

    l, r = 0, 0
    while l < len(data):
        r = min(l + batch_size, len(data))
        batch = defaultdict(list)

        for i in data_idx[l : r]:
            ns = len(data[i]['text'])
            null_sents = 0
            for j in range(ns):
                sent = data[i]['text'][j]
                if len(sent) == 0:
                    null_sents += 1
                    continue
                batch['fact_text'].append(torch.tensor(sent, dtype=torch.long, device=device))
                batch['sent_lens'].append(len(sent))

                if bsl:  batch['sent_labels'].append(torch.tensor(data[i]['sent_labels'][j], dtype=torch.float, device=device))

            if bdl:  batch['doc_labels'].append(torch.tensor(data[i]['doc_labels'], dtype=torch.float, device=device))
            batch['doc_lens'].append(ns - null_sents)

        batch['fact_text'] = U.pad_sequence(batch['fact_text'], batch_first=True)
        batch['sent_lens'] = torch.tensor(batch['sent_lens'], dtype=torch.long, device=device)
        batch['doc_lens'] = torch.tensor(batch['doc_lens'], dtype=torch.long, device=device)

        if bsl:  batch['sent_labels'] = torch.stack(batch['sent_labels'])
        if bdl:  batch['doc_labels'] = torch.stack(batch['doc_labels'])

        yield batch
        l = r

    return


def calc_metrics(tracker):
    metrics = {}
    precision = tracker['match'] / tracker['preds']
    recall = tracker['match'] / tracker['labels']
    f1 = 2 * precision * recall / (precision + recall)

    precision[torch.isnan(precision)] = 0
    recall[torch.isnan(recall)] = 0
    f1[torch.isnan(f1)] = 0

    metrics['label-P'] = precision.tolist()
    metrics['label-R'] = recall.tolist()
    metrics['label-F1'] = f1.tolist()

    metrics['macro-P'] = precision.mean().item()
    metrics['macro-R'] = recall.mean().item()
    metrics['macro-F1'] = f1.mean().item()

    return metrics
