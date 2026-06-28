import torch
import numpy as np

def _calculate_metrics(tp, tn, fp, fn, eps=1e-7):
    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    return iou, dice, sensitivity, specificity

def test_func(model, val_loader, device, threshold=0.5, eps=1e-7):
    model.to(device)
    model.eval()

    all_ious = []
    all_dices = []
    all_sensitivities = []
    all_specificities = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            if outputs.shape[1] > 1: # Multi-class
                probs = torch.softmax(outputs, dim=1)[:, 1, :, :]
            else: # Binary
                probs = torch.sigmoid(outputs).squeeze(1)

            true_mask_batch = (labels.squeeze(1) > 0.5).bool()
            pred_mask_batch = (probs > threshold).bool()


            batch_size = images.shape[0]
            for i in range(batch_size):
                p_mask = pred_mask_batch[i]
                t_mask = true_mask_batch[i]

                tp = (p_mask & t_mask).sum().item()
                tn = (~p_mask & ~t_mask).sum().item()
                fp = (p_mask & ~t_mask).sum().item()
                fn = (~p_mask & t_mask).sum().item()

                iou, dice, sens, spec = _calculate_metrics(tp, tn, fp, fn, eps)
                
                all_ious.append(iou)
                all_dices.append(dice)
                all_sensitivities.append(sens)
                all_specificities.append(spec)


    results = {
        "IoU": (np.mean(all_ious), np.std(all_ious)),
        "Dice": (np.mean(all_dices), np.std(all_dices)),
        "Sensitivity": (np.mean(all_sensitivities), np.std(all_sensitivities)),
        "Specificity": (np.mean(all_specificities), np.std(all_specificities))
    }

    print(f"{'Metric':<15} | {'Mean':<10} | {'Std':<10}")
    print("-" * 40)
    for metric, (m, s) in results.items():
        print(f"{metric:<15} | {m:.4f}     | {s:.4f}")

    return results