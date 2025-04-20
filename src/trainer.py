import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    return model.validation_epoch_end([model.validation_step(batch) for batch in val_loader])

def fit(num_epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        train_accs = []

        for batch in train_loader:
            loss_dict = model.training_step(batch)
            loss, acc = loss_dict['train_loss'], loss_dict['train_acc']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss)
            train_accs.append(acc)

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['train_acc'] = torch.stack(train_accs).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history

def plot_metrics(history, class_names=None):
    train_losses = [x['train_loss'] for x in history]
    val_losses = [x['val_loss'] for x in history]
    train_accs = [x['train_acc'] for x in history]
    val_accs = [x['val_acc'] for x in history]
    val_recalls = [x['val_recalls'] for x in history]  # list of list

    num_classes = len(val_recalls[0])
    class_names = class_names or [f"Class {i}" for i in range(num_classes)]

    # Plot Losses
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracies
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Recall per Class
    plt.subplot(1, 3, 3)
    for class_idx in range(num_classes):
        recalls = [epoch[class_idx] for epoch in val_recalls]
        plt.plot(recalls, label=class_names[class_idx])
    
    plt.title('Recall per Class')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()

