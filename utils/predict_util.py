import torch
from utils.train_util import prepare_mini

def infer(model, data, label_vocab, batch_size=5, device='cuda'):
    model.eval()
    predictions = []

    inv_label_vocab = {i:l for l, i in label_vocab.items()}

    for batch in prepare_mini(data, batch_size, device, False):
        if 'cuda' in device:
            torch.cuda.empty_cache()
        
        model_out = model(batch)

        for doc in model_out['doc_preds']:
            pred = [inv_label_vocab[idx.item()] for idx in doc.nonzero(as_tuple=False)]
            predictions.append(pred)

    return predictions
