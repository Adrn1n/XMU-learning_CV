import torch
from torch import accelerator, Generator, hub
from torch.utils.data import DataLoader, random_split, Subset
import time
import os
from tqdm import tqdm
from torchvision import datasets, models, transforms
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from typing import cast, Sized
from sklearn.metrics import confusion_matrix

DEVICE = (
    cast(str | torch.device | int, accelerator.current_accelerator())
    if accelerator.is_available()
    else torch.device("cpu")
)
MAX_EPOCHS_PRINT = 100
MAX_MODEL_SAVE = 5
SEED = 42


class MLRunner:
    def __init__(
        self,
        criterion=None,
        dataset=None,
        device=torch.device("cpu"),
        model=None,
        opt=None,
        score_fn=None,
        seed=SEED,
        *args,
        **kwargs,
    ):
        self.criterion = criterion
        self.dataset = dataset
        self.device = device
        self.model = model
        self.opt = opt
        self.score_fn = score_fn
        self.seed = seed
        if self.dataset is None:
            self.dataloader = None
        else:
            self.dataloader = DataLoader(self.dataset, *args, **kwargs)

    @staticmethod
    def get_predicts(ys):
        return ys

    def set_criterion(self, criterion):
        self.criterion = criterion
        return self.criterion

    def set_dataset(self, dataset):
        self.dataset = dataset
        return self.dataset

    def set_dataloader(self, *args, **kwargs):
        self.dataloader = DataLoader(self.dataset, *args, **kwargs)
        return self.dataloader

    def set_model(self, model):
        self.model = model
        return self.model

    def set_opt(self, opt, *args, **kwargs):
        self.opt = opt(
            filter(lambda p: p.requires_grad, self.model.parameters()), *args, **kwargs
        )
        return self.opt

    def fit(
        self, save_dir=None, is_train=False, scheduler=None, epochs=1, *args, **kwargs
    ):
        assert self.dataloader is not None
        since = time.time()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        ls, preds, scores = [], [], []
        if is_train:
            if scheduler is not None:
                scheduler = scheduler(self.opt, *args, **kwargs)
            self.model.train()
        else:
            self.model.eval()
        for ep in range(epochs):
            ls.append(0)
            if not is_train:
                preds.append([])
                scores.append(0)
            for x, y in tqdm(self.dataloader, desc=f"Epoch {ep+1}/{epochs}"):
                x, y_device = x.to(self.device), y.to(self.device)
                with torch.set_grad_enabled(is_train):
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y_device)
                    if is_train:
                        self.opt.zero_grad()
                        loss.backward()
                        self.opt.step()
                    else:
                        ps = self.get_predicts(outputs)
                        preds[-1].append(ps)
                        scores[-1] += self.score_fn(ps, y).item()
                ls[-1] += loss.item() * x.size(0)
            ls[-1] /= len(cast(Sized, cast(object, self.dataloader.dataset)))
            if (epochs <= MAX_EPOCHS_PRINT) or (
                (ep + 1) % (epochs // MAX_EPOCHS_PRINT) == 0
            ):
                print(f"Epoch {ep+1}/{epochs}, Loss: {ls[-1]:.6f}")
            if is_train:
                if save_dir is not None and (
                    (epochs <= MAX_MODEL_SAVE)
                    or ((ep + 1) % (epochs // MAX_MODEL_SAVE) == 0)
                ):
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(save_dir, f"model_epoch{ep+1}.pth"),
                    )
                if scheduler is not None:
                    scheduler.step()
            else:
                preds[-1] = torch.cat(preds[-1])
        time_elapsed = time.time() - since
        if is_train:
            return ls, time_elapsed
        else:
            return ls, preds, scores, time_elapsed


class ClassificationRunner(MLRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_fn = self.get_score

    def get_predicts(self, ys):
        ys = ys.cpu()
        mask = ys == torch.max(ys, 1, True).values
        ids = torch.where(mask)[1]
        cnts = mask.sum(1)
        offsets = (
            torch.randint(
                0,
                ys.size(1),
                (ys.size(0),),
                generator=Generator().manual_seed(self.seed),
            )
            % cnts
        )
        starts = torch.cat((torch.tensor([0]), torch.cumsum(cnts, dim=0)[:-1]))
        return ids[starts + offsets]

    @staticmethod
    def get_score(preds, ys):
        return (preds == ys).sum()


RUNNER = ClassificationRunner(device=DEVICE)
MODEL_DIR = "models"
MODEL_PARAMS = {"weights": "IMAGENET1K_V1"}
data_dir, resize, crop = "data/15-Scene", 256, 224
d_s = datasets.ImageFolder(
    data_dir,
    transform=transforms.Compose(
        [transforms.Resize(resize), transforms.CenterCrop(crop), transforms.ToTensor()]
    ),
)
CLS_NAMES = d_s.classes
hub.set_dir(MODEL_DIR)


def build_model(freeze):
    model = models.resnet50(**MODEL_PARAMS)
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLS_NAMES))
    return model


def plot_loss(losses, name, save_path=None):
    plt.plot(losses, label="loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} Training Loss")
    plt.grid()
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


CM_TEXT_TH_F = 0.5


def plot_confusion_matrix(cm, classes, name, save_path=None):
    plt.imshow(cm, interpolation="nearest")
    thresh = cm.max() * CM_TEXT_TH_F
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="black" if cm[i, j] > thresh else "white",
            )
    marks = np.arange(len(classes))
    plt.xticks(marks, classes)
    plt.gca().xaxis.tick_top()
    plt.yticks(marks, classes)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.gca().xaxis.set_label_position("top")
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


CRITERION = nn.CrossEntropyLoss()
t_p = 0.5
TRAIN_SET, VAL_SET = random_split(d_s, [t_p, 1 - t_p], Generator().manual_seed(SEED))
mean, std = (torch.zeros(3) for _ in range(2))
n = 0
for img, _ in DataLoader(TRAIN_SET, batch_size=1000):
    n += img.size(0) * img.size(2) * img.size(3)
    mean += img.sum(dim=[0, 2, 3])
    std += (img**2).sum(dim=[0, 2, 3])
mean /= n
std = (std / n - mean**2).sqrt()
TRAIN_SET = Subset(
    datasets.ImageFolder(
        data_dir,
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    ),
    TRAIN_SET.indices,
)
BATCH_SIZE = 4
OPT = optim.Adam
OPT_PARAMS = {"lr": 1e-3}
OUTPATH = "outputs"
SCHEDULER = lr_scheduler.StepLR
EPOCHS = 25
SCHE_PARAMS = {"step_size": 7, "gamma": 0.1}
IMG_PATH = "imgs"
VAL_SET = Subset(
    datasets.ImageFolder(
        data_dir,
        transform=transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(crop),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    ),
    VAL_SET.indices,
)


def run_model(runner, model, name):
    runner.set_criterion(CRITERION)
    runner.set_dataset(TRAIN_SET)
    runner.set_dataloader(batch_size=BATCH_SIZE, shuffle=True)
    runner.set_model(model)
    runner.set_opt(OPT, **OPT_PARAMS)
    train_ls, _ = runner.fit(
        f"{OUTPATH}/{MODEL_DIR}/{name}", True, SCHEDULER, EPOCHS, **SCHE_PARAMS
    )
    os.makedirs(f"{OUTPATH}/{IMG_PATH}", exist_ok=True)
    plot_loss(train_ls, name, save_path=f"{OUTPATH}/{IMG_PATH}/loss_{name}.png")
    runner.set_dataset(VAL_SET)
    runner.set_dataloader(batch_size=BATCH_SIZE, shuffle=False)
    _, preds, _, _ = runner.fit(is_train=False)
    plot_confusion_matrix(
        confusion_matrix(
            [
                cast(datasets.ImageFolder, VAL_SET.dataset).targets[i]
                for i in VAL_SET.indices
            ],
            preds[-1].numpy(),
        ),
        CLS_NAMES,
        name,
        save_path=f"{OUTPATH}/{IMG_PATH}/cm_{name}.png",
    )


if __name__ == "__main__":
    run_model(RUNNER, build_model(freeze=False).to(DEVICE), "ResNet50_FineTune")
    run_model(RUNNER, build_model(freeze=True).to(DEVICE), "ResNet50_LastLayer")
