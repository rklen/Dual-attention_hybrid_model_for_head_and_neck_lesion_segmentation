def dice_loss(pred, target, smooth=1e-5):

    # Apply softmax to get probabilities
    pred_probs = torch.sigmoid(pred)  # shape (B, 2, H, W)
    
    # Take the foreground class 
    # pred_fg = pred_probs[:, 1, :, :]  # shape (B, H, W)


    # Flatten
    pred_fg_flat = pred_probs.contiguous().view(pred_probs.size(0), -1)  # (B, H*W)
    target_flat = target.contiguous().view(target.size(0), -1)     # (B, H*W)

    # Compute Dice
    intersection = (pred_fg_flat * target_flat).sum(1)
    union = pred_fg_flat.sum(1) + target_flat.sum(1)

    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice

    return dice_loss.mean()