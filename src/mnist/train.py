import wandb

from mnist.config import load_default_config
from mnist.data import get_dataloaders
from mnist.utils import set_seed
from mnist.model import LeNet5
import torch

def main():
    model = LeNet5()
    config = load_default_config()
    data = get_dataloaders(config.data)
    print(data)
    raise Exception("break to test data loading only")

    set_seed(config.train.seed)
    logits = model()

    print("LOGITS")
    print(logits)

    #display(pd.DataFrame(logits.detach().numpy(), columns=range(10)).round(3))
    df = pd.DataFrame(logits.detach().numpy(), columns=range(10)).round(3)
    display(df.style.highlight_max(axis=1, color="green"))

    loss_fn = nn.CrossEntropyLoss()
    labels = torch.randint(0, 10, (NUM_SAMPLES,))
    loss = loss_fn(logits, labels)
    print("LOSS")
    print(loss)

    print("PREDICTIONS")
    preds = logits.argmax(dim=1)
    print(preds)

    softmaxes = torch.softmax(logits, dim=1)
    print("SOFTMAXES")
    print(softmaxes)

    """
    run = wandb.init(project="mnist-smoke-test")
    wandb.log({"train/loss": 0})
    run.summary["final_val_acc"] = 1
    wandb.finish()
    print("Hello from mnist!")
    """


if __name__ == "__main__":
    main()
