import config
import torch


def test_loop(dataloader, model, loss_fn):
    """
    Evaluates the model on the provided dataloader.
    Args:
        dataloader: The dataloader containing the validation data.
        model: The model to be evaluated.
        loss_fn: The loss function to use for evaluation.
    """
    model.eval()
    num_batches = len(dataloader)
    validation_loss = 0
    validation_correct = 0
    with torch.no_grad():
        for image_batch, labels in dataloader:
            image_batch, labels = (
                image_batch.to(config.DEVICE),
                labels.to(config.DEVICE),
            )
            pred = model(image_batch)
            validation_loss += loss_fn(pred, labels).item()
            validation_correct += (
                (pred.argmax(1) == labels).type(torch.float).sum().item()
            )

    validation_loss /= num_batches
    validation_accuracy = 100 * validation_correct / len(dataloader.dataset)

    print(
        f"Validation Result: \n Accuracy: {validation_accuracy:.2f}%, Avg loss: {validation_loss:.2f}"
    )
