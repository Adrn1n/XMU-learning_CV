from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary
from sklearn.metrics import confusion_matrix

DATA_DIR = "data/"
data = torch.stack(
    [
        img
        for img, _ in datasets.CIFAR10(
            DATA_DIR, True, transforms.ToTensor(), download=True
        )
    ],
    dim=0,
)
data_mean = data.mean(dim=(0, 2, 3))
data_std = data.std(dim=(0, 2, 3))
MAX_EPOCHS_PRINT = 100


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()
OPT = optim.Adam
EPOCHS = 500
TRAIN_SET = datasets.CIFAR10(
    DATA_DIR,
    True,
    transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std),
        ]
    ),
    download=True,
)
BATCH_SIZE = 512
CRITERION = nn.CrossEntropyLoss()
LR = 5e-5
TEST_SET = datasets.CIFAR10(
    DATA_DIR,
    False,
    transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(data_mean, data_std)]
    ),
    download=True,
)
img_size = (32, 32, 3)
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_mlp(dims, activations, dropouts, flatten=True):
    layers = [nn.Flatten()] if flatten else []
    if dims:
        max_ai = len(activations) - 1
        max_di = len(dropouts) - 1
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), activations[min(i, max_ai)]]
            if dropouts[min(i, max_di)]:
                layers.append(nn.Dropout(dropouts[min(i, max_di)]))
    return layers


mlp_dims = [img_size[0] * img_size[1] * img_size[2], 1024, 512, 256, len(classes)]
mlp_activations = [nn.ReLU()] * 3 + [nn.Identity()]
mlp_dropouts = [0.5, 0.5, 0.3, None]
MLP_NET = nn.Sequential(*build_mlp(mlp_dims, mlp_activations, mlp_dropouts))


def build_cnn(
    channels,
    kernel_sizes,
    paddings,
    activations,
    poolings,
    cls_dims,
    cls_activations,
    cls_dropouts,
):
    layers = []
    if channels:
        max_ki, max_pai, max_ai, max_poi = (
            len(kernel_sizes) - 1,
            len(paddings) - 1,
            len(activations) - 1,
            len(poolings) - 1,
        )
        for i in range(len(channels) - 1):
            layers += [
                nn.Conv2d(
                    channels[i],
                    channels[i + 1],
                    kernel_sizes[min(i, max_ki)],
                    padding=paddings[min(i, max_pai)],
                ),
                activations[min(i, max_ai)],
            ]
            if poolings[min(i, max_poi)]:
                layers.append(poolings[min(i, max_poi)])
        layers += [
            nn.Flatten(),
            nn.LazyLinear(cls_dims[0]),
            cls_activations[0],
            nn.Dropout(cls_dropouts[0]),
        ]
        if len(cls_dims) > 1:
            layers += build_mlp(
                cls_dims[0:], cls_activations[1:], cls_dropouts[1:], False
            )
    return layers


cnn_channels = [img_size[2], 32, 64, 128]
cnn_kernel_sizes = [3, 5, 3]
cnn_padding_sizes = [2, 3, 1]
cnn_activations = [nn.ReLU()]
cnn_poolings = [None, None, nn.MaxPool2d(2)]
cnn_mlp_dims = [64, len(classes)]
cnn_mlp_activations = [nn.ReLU(), nn.Identity()]
cnn_mlp_dropouts = [0.5, None]
CNN_NET = nn.Sequential(
    *build_cnn(
        cnn_channels,
        cnn_kernel_sizes,
        cnn_padding_sizes,
        cnn_activations,
        cnn_poolings,
        cnn_mlp_dims,
        cnn_mlp_activations,
        cnn_mlp_dropouts,
    )
)


class ImgClsNN(nn.Module):
    def __init__(self, net, device=torch.device("cpu")):
        super().__init__()
        self.net = net
        self.device = device
        self.to(device)

    def fit(self, opt, epochs, data_loader, criterion, *args, **kwargs):
        optimizer = opt(self.parameters(), *args, **kwargs)
        ls = []
        self.train()
        for ep in range(epochs):
            ls.append(0)
            for x, y in tqdm(data_loader, desc=f"Epoch {ep+1}/{epochs}"):
                optimizer.zero_grad()
                loss = criterion(self(x.to(self.device)), y.to(self.device))
                loss.backward()
                optimizer.step()
                ls[-1] += loss.item() * x.size(0)
            ls[-1] /= len(data_loader.dataset)
            if (epochs <= MAX_EPOCHS_PRINT) or (
                (ep + 1) % (epochs // MAX_EPOCHS_PRINT) == 0
            ):
                print(f"Epoch {ep+1}/{epochs}, Loss: {ls[-1]:.6f}")
        return ls

    def forward(self, x):
        return self.net(x.to(self.device))

    @staticmethod
    def get_data_loader(data_set, *args, **kwargs):
        return DataLoader(data_set, *args, **kwargs)

    @staticmethod
    def get_predicts(ys):
        ys = ys.cpu()
        mask = ys == torch.max(ys, 1, True).values
        ids = torch.where(mask)[1]
        cnts = mask.sum(1)
        offsets = torch.randint(0, ys.size(1), (ys.size(0),)) % cnts
        starts = torch.cat((torch.tensor([0]), torch.cumsum(cnts, dim=0)[:-1]))
        return ids[starts + offsets]

    def evaluate(self, data_loader, criterion):
        correct, loss = 0, 0
        preds = []
        self.eval()
        with torch.no_grad():
            for x, y in tqdm(data_loader, desc="Evaluating"):
                x, y_device = x.to(self.device), y.to(self.device)
                outputs = self(x)
                loss += criterion(outputs, y_device).item() * x.size(0)
                pred = self.get_predicts(outputs)
                correct += (pred == y).sum().item()
                preds.append(pred)
        return correct, torch.cat(preds), loss / len(data_loader.dataset)


def run_model(net, name):
    model = ImgClsNN(net, DEVICE)
    summary(model, (3, 32, 32), device=DEVICE.type if DEVICE.type == "cuda" else "cpu")
    _ = model.fit(
        OPT,
        EPOCHS,
        model.get_data_loader(TRAIN_SET, batch_size=BATCH_SIZE, shuffle=True),
        CRITERION,
        lr=LR,
    )
    correct, preds, _ = model.evaluate(
        model.get_data_loader(TEST_SET, batch_size=BATCH_SIZE), CRITERION
    )
    print(f"{name} Test Accuracy: {correct / len(TEST_SET):.6f}")
    print(
        f"{name} Confusion Matrix:\n",
        confusion_matrix(TEST_SET.targets, preds.numpy()),
    )


if __name__ == "__main__":
    run_model(MLP_NET, "MLP")
    run_model(CNN_NET, "CNN")
