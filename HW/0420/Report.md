# Report 4
## Requirements
- Use the PyTorch deep learning framework to build a multi-layer fully connected neural network and a convolutional neural network for classifying CIFAR-10 data
- Manually calculate the input and output sizes of each layer of the model, as well as the corresponding number of parameters; compare with the results given by torchsummary
- Compare the accuracy of the two network models and provide a confusion matrix for the test set

## Implementation
### Methodology
Both the multi-layer fully connected neural network (MLP) and the convolutional neural network (CNN) are neural networks designed for image classification tasks, and to compare their performance on the CIFAR-10 dataset, a shared framework is implemented using PyTorch. This framework handles data loading, model training, prediction, and evaluation, including accuracy calculation and confusion matrix generation. The two models share the same pipeline, while their architectures differ, which can be altered by changing the input parameters of the framework.

To improve the model's generalization ability, the dataset is preprocessed by calculating the mean and standard deviation of the training data for normalization. In addition, data augmentation techniques such as random horizontal flipping and random cropping are applied to the training set.

To facilitate the construction of the MLP and CNN architectures, two helper functions are implemented. These functions take in parameters such as layer dimensions, activation functions, dropout rates, kernel sizes, and pooling layers to construct the respective network architectures.

### Overview
#### MLP
- First layer: A linear layer that takes the flattened input image($32 \times 32 \times 3 = 3072$), and outputs 1024 features, followed by a ReLU activation function and a dropout layer with a dropout rate of 0.5. The number of parameters for this layer is calculated as $3072 \times 1024 + 1024 = 3146752$.
- Second layer: A linear layer that takes the 1024 features from the previous layer and outputs 512 features, followed by a ReLU activation function and a dropout layer with a dropout rate of 0.5. The number of parameters for this layer is calculated as $1024 \times 512 + 512 = 524800$.
- Third layer: A linear layer that takes the 512 features from the previous layer and outputs 256 features, followed by a ReLU activation function and a dropout layer with a dropout rate of 0.3. The number of parameters for this layer is calculated as $512 \times 256 + 256 = 131328$.
- Output layer: A linear layer that takes the 256 features from the previous layer and outputs 10 features corresponding to the 10 classes in the CIFAR-10 dataset. The number of parameters for this layer is calculated as $256 \times 10 + 10 = 2570$.

#### CNN
- Feature extraction layers:
  - First convolutional layer: A convolutional layer that takes the input image with 3 channels and $32 \times 32$ spatial dimensions, and outputs 32 feature maps using a kernel size of 3 and padding of 2, followed by a ReLU activation function, and no pooling. Therefore the output is $32 \times (32 + 2 \times 2 - 3 + 1) \times (32 + 2 \times 2 - 3 + 1) = 32 \times 34 \times 34$. The number of parameters for this layer is calculated as $3 \times 32 \times 3 \times 3 + 32 = 896$.
  - Second convolutional layer: A convolutional layer that takes the 32 feature maps from the previous layer and outputs 64 feature maps using a kernel size of 5 and padding of 3, followed by a ReLU activation function, and no pooling. Therefore the output is $64 \times (34 + 2 \times 3 - 5 + 1) \times (34 + 2 \times 3 - 5 + 1) = 64 \times 36 \times 36$. The number of parameters for this layer is calculated as $32 \times 64 \times 5 \times 5 + 64 = 51264$.
  - Third convolutional layer: A convolutional layer that takes the 64 feature maps from the previous layer and outputs 128 feature maps using a kernel size of 3 and padding of 1, followed by a ReLU activation function, and a max pooling layer with a kernel size of 2. Therefore the output is $128 \times [\frac{36 + 2 \times 1 - 3 + 1}{2}] \times [\frac{36 + 2 \times 1 - 3 + 1}{2}] = 128 \times 18 \times 18$. The number of parameters for this layer is calculated as $64 \times 128 \times 3 \times 3 + 128 = 73856$.
- Classification layers:
  - First linear layer: A linear layer that takes the flattened output from the feature extraction layers (which has $128 \times 18 \times 18 = 41472$ features) and outputs 64 features, followed by a ReLU activation function and a dropout layer with a dropout rate of 0.5. The number of parameters for this layer is calculated as $41472 \times 64 + 64 = 2654272$.
  - Output layer: A linear layer that takes the 64 features from the previous layer and outputs 10 features corresponding to the 10 classes in the CIFAR-10 dataset. The number of parameters for this layer is calculated as $64 \times 10 + 10 = 650$.

### Parameters
- `DATA_DIR`: The directory where the CIFAR-10 dataset is stored or will be downloaded to
- `MAX_EPOCHS_PRINT`: The maximum number of epochs to print during training
- `DEVICE`: The device to run the model on
- `OPT`: The optimizer to use
- `EPOCHS`: The number of epochs to train the model
- `TRAIN_SET`: The training dataset with transformations applied
- `BATCH_SIZE`: The batch size for training and evaluation
- `CRITERION`: The loss function to use
- `LR`: The learning rate for the optimizer
- `TEST_SET`: The test dataset with transformations applied
- `MLP_NET`: The multi-layer fully connected neural network architecture
- `CNN_NET`: The convolutional neural network architecture

### Features
- If the number of epochs is less than or equal to `MAX_EPOCHS_PRINT`, the loss will be printed for every epoch. Otherwise, the loss will be printed for every `epochs // MAX_EPOCHS_PRINT` epochs.
- The device is automatically selected based on availability (CUDA first, then XPU, then MPS, and finally CPU).
- The number of layers for MLP is determined by the length of the input `dims` list (every input dimension and the last layer's output dimension), and every layer must have a corresponding activation function (can be `nn.Identity()` if no activation is desired) but dropout is optional (can be `None`). The length of `activations` can be less than the required length, in which case the last activation function will be used for the remaining layers. The same applies to `dropouts`.
- The number of convolutional layers for CNN is determined by the length of the input `channels` list (every input channel and the last layer's output channel), and every layer must have a corresponding kernel size, padding (can be `0` if no padding is desired), and activation function (can be `nn.Identity()` if no activation is desired), but pooling is optional (can be `None`). The length of `kernel_sizes`, `paddings`, `activations`, and `poolings` can be less than the required length, in which case the last value will be used for the remaining layers. The classification layers are constructed the same way as MLP, with the same rules for activations and dropouts. But at least one classification layer is required. And the first layer's input dimension is omitted and its dropout must be specified (can't be `None`, but can be `0` if no dropout is desired).
- If multiple classes have the same maximum output value during prediction, one of them will be randomly selected as the predicted class to avoid bias.

## Code
```python
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

```

## Results
### MLP
```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
           Flatten-1                 [-1, 3072]               0
            Linear-2                 [-1, 1024]       3,146,752
              ReLU-3                 [-1, 1024]               0
           Dropout-4                 [-1, 1024]               0
            Linear-5                  [-1, 512]         524,800
              ReLU-6                  [-1, 512]               0
           Dropout-7                  [-1, 512]               0
            Linear-8                  [-1, 256]         131,328
              ReLU-9                  [-1, 256]               0
          Dropout-10                  [-1, 256]               0
           Linear-11                   [-1, 10]           2,570
         Identity-12                   [-1, 10]               0
================================================================
Total params: 3,805,450
Trainable params: 3,805,450
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.06
Params size (MB): 14.52
Estimated Total Size (MB): 14.59
----------------------------------------------------------------

MLP Test Accuracy: 0.598600
MLP Confusion Matrix:
 [[630  35  30  17  43   5  23  21 113  83]
 [ 17 672   6  12   6   3  18   6  43 217]
 [ 77  16 358  94 122  79 122  82  16  34]
 [ 22  13  43 439  68 184 103  61  18  49]
 [ 36   8  71  70 483  44 123 127  20  18]
 [  8   7  39 240  61 473  46  88  15  23]
 [  9  16  35  83  50  29 723  18  10  27]
 [ 13   6  17  51  76  46  17 721  10  43]
 [ 75  57   4  20  25   5   6   6 731  71]
 [ 27 100   8  17  11   6  15  22  38 756]]
```

### CNN
```text
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 34, 34]             896
              ReLU-2           [-1, 32, 34, 34]               0
            Conv2d-3           [-1, 64, 36, 36]          51,264
              ReLU-4           [-1, 64, 36, 36]               0
            Conv2d-5          [-1, 128, 36, 36]          73,856
              ReLU-6          [-1, 128, 36, 36]               0
         MaxPool2d-7          [-1, 128, 18, 18]               0
           Flatten-8                [-1, 41472]               0
            Linear-9                   [-1, 64]       2,654,272
             ReLU-10                   [-1, 64]               0
          Dropout-11                   [-1, 64]               0
           Linear-12                   [-1, 10]             650
         Identity-13                   [-1, 10]               0
================================================================
Total params: 2,780,938
Trainable params: 2,780,938
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 5.00
Params size (MB): 10.61
Estimated Total Size (MB): 15.62
----------------------------------------------------------------

CNN Test Accuracy: 0.786700
CNN Confusion Matrix:
 [[822  25  35  14  12   4   5   8  54  21]
 [ 14 898   3   7   1   2   3   0  12  60]
 [ 55   1 653  67  77  53  60  19   8   7]
 [ 13   4  46 610  34 203  49  15  14  12]
 [ 13   3  58  59 716  41  47  49  13   1]
 [  7   3  35 138  34 750  15  15   0   3]
 [  4   3  42  61  13  20 849   5   1   2]
 [ 14   1  24  24  36  76   5 810   0  10]
 [ 50  20   4   5   4   5   3   1 893  15]
 [ 21  54   3   9   1   3   5   9  29 866]]
```
