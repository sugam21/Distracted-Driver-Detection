import config
import torch


def test_loop(dataloader, model, loss_fn):
    # set the model to eval mode - imp for batch norma and dropout
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # evaluate the model with torch.no_grad() ensures that no gradients are
    # computed during test mode
    # also reduce unnecessary gradients computations and memory usage for
    # tensors with requires_grad = True
    with torch.no_grad():
        for image_batch, labels in dataloader:
            (image_batch, labels) = (
                image_batch.to(config.DEVICE),
                labels.to(config.DEVICE),
            )
            pred = model(image_batch)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Validation Error : \n Accuracy: {100 * correct:0.1f}%, Avg loss: {test_loss:>8f} \n"
    )
