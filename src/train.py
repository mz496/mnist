import wandb


def main():
    run = wandb.init(project="mnist-smoke-test")
    wandb.log({"train/loss": 0})
    run.summary["final_val_acc"] = 1
    wandb.finish()
    print("Hello from mnist!")


if __name__ == "__main__":
    main()
