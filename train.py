



import os
import torch
from tqdm import tqdm
import copy
from models.ourmodel import mymodel
from utils.loss import dice_loss
from utils.metrics import test_func
base_lr = 0.0001
optimizer = torch.optim.Adam(mymodel.parameters(), lr=base_lr)

epochs = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def training_function(
    train_loader,
    val_loader,
    model,
    device,
    optimizer,
    epochs,
    criterion,
    save_path="swinT_model.pth",
    patience=5,
    minimize=False,   
    verbose=True
):


    model.to(device)
    best_metric = float("inf") if minimize else -float("inf")
    best_state = None
    epochs_no_improve = 0
    comparator = (lambda a, b: a < b) if minimize else (lambda a, b: a > b)

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()

            images = images.to(device)
            labels = labels.to(device)
            labels = labels.float()  # keep your existing preprocessing

            outs = model(images)

            dice = dice_loss(outs[0], labels) + 0.3*dice_loss(outs[1], labels) 
            loss = dice 

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / (len(train_loader.dataset) if hasattr(train_loader, "dataset") else len(train_loader))
        # Validation
        model.eval()
        with torch.no_grad():
            result = test_func(model, val_loader, device)  # your IoU function; higher is better for IoU
            val_metric = result['IoU'][0]
        if verbose:
            print(f"Epoch {epoch:03d} | Train loss: {epoch_train_loss:.4f} | Val metric: {result['IoU'][0]:.4f} ")

        # Check for improvement
        if comparator(val_metric, best_metric):
            best_metric = val_metric
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric
            }
            # ensure directory exists
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(best_state, save_path)
            epochs_no_improve = 0
            
            if verbose:
                print(f"  -> Improved. Saved checkpoint to {save_path}")
        else:
            epochs_no_improve += 1

            if verbose:
                print(f"  -> No improvement for {epochs_no_improve}/{patience} epochs")

        # Early stopping
        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping triggered. No improvement in {patience} epochs.")
            break