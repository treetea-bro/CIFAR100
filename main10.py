from __future__ import print_function
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
from utils import progress_bar
from efficientnet_pytorch import EfficientNet
from cutout import Cutout


def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR100 Training")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--gamma", default=0.1, type=float, help="lr decay")
    parser.add_argument("--wd", default=1e-6, type=float, help="weights decay")
    parser.add_argument("--ne", default=30, type=int, help="number of epochs")
    parser.add_argument("--nsc", default=10, type=int, help="number of step for a lr")
    parser.add_argument(
        "--batch_split", default=1, type=int, help="spliting factor for the batch"
    )
    parser.add_argument("--batch", default=32, type=int, help="size of the batch")
    parser.add_argument(
        "--alpha",
        default=0.1,
        type=float,
        help="mixup interpolation coefficient (default: 1)",
    )

    args = parser.parse_args()

    # 시드 설정으로 다음에도 같은 accuracy 결과가 나올 수 있도록 함
    torch.manual_seed(0)
    np.random.seed(0)

    # 현재PC에서 GPU 이용가능 여부 확인해서 변수에 이용가능한 GPU 종류담기
    # apple silicon이면 mps를 사용하고, 아니면 cuda를 이용하고, cuda도 안되면 cpu 이용
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("device : ", device)

    mini_batch_size = args.batch // args.batch_split

    #  각 이미지에 적용할 변형기법을 파이프라인으로 정의
    #  이미지는 각 함수를 통과하여 변형됨
    transform_train = transforms.Compose(
        [
            # 보간 기법으로 BILINEAR를 사용하여 160x160사이즈로 이미지를 변환시킨다.
            # transforms.Resize(160),
            # default 값은 0.5이며, 50% 확률로 이미지 좌우를 반전시킨다.
            transforms.RandomHorizontalFlip(),
            # 이미지를 파이토치의 텐서 객체로 변환하여 trainable하게 변환시킨다.
            transforms.ToTensor(),
            # 이미지를 정규화하여 학습시에 더 빠르고 안정적으로 weights를 변화시켜간다.
            # 위에서 텐서 객체로 변환할 때 0-255범위를 0-1범위로 바꾸기 때문에
            # ToTensor 함수 다음에 Normalize 함수를 적용시켜야한다.
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            Cutout(),
        ]
    )

    # 테스트 데이터에 대한 변형기법 파이프라인을 설정
    transform_test = transforms.Compose(
        [
            # 보간 기법으로 BILINEAR를 사용하여 200x200사이즈로 이미지를 변환시킨다.
            # transforms.Resize(200),
            transforms.ToTensor(),
            # 학습때와 같이 정규화를 진행하여 같은 환경을 만들어 줌으로 써
            # 더 정답을 잘 맞출 수 있도록 한다.
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    # CIFAR100 데이터를 trainset과 testset으로 나눠서 data폴더에 다운로드 받는다.
    # 1. 폴더에 이미지데이터가 있으면 메모리에 이미지 데이터를 로드한다. 없으면 다운로드 받고 로드한다.
    # 2. transform 파이프라인을 붙여서 나중에 DataLoader가 데이터를 로드할 때
    # 각 이미지별로 하나씩 transform 파이프라인을 적용한다.
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    # 정답 라벨과 모델이 예측한 값의 오차를 측정하는 손실함수
    # 손실함수로 구한 오차를 토대로 나중에 옵티마이저로 weights를 업데이트 한다.
    criterion = nn.CrossEntropyLoss()

    def mixup_data(x: Tensor, y, alpha=1.0, lam=1.0, count=0):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if count == 0:
            if alpha > 0:
                # 현재 args 값에 alpha는 0.1로 설정되어있어서
                # lam의 값은 0이나 1에 가까운 값이 난수로 생성되게 된다.
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.0

        # 매 미니배치(32)마다 이 함수가 호출되는데, 그때마다
        # 순서가 0-31의 인덱스 이미지와 0-31범위에 해당하긴하지만
        # 순서가 랜덤으로 생성된 이미지 ex: [1,4,6,8,0,2 ...]
        # 를 mixup 하기위해 batch_size(32)를 인자로 랜덤순열을 생성한다.
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)

        # 위에서 난수로 구한 lam값과 기존 배치 0-31 인덱스와
        # 랜덤순열로 구한 인덱스를 가지고 이미지들과 라벨들을 mix한다.
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # Training
    def train(epoch, trainloader):
        print("\nEpoch: %d" % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        count = 0
        lam = 1.0
        optimizer.zero_grad()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if count == args.batch_split:
                optimizer.step()
                optimizer.zero_grad()
                count = 0
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, args.alpha, lam, count
            )
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss = loss / args.batch_split
            loss.backward()
            count += 1
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (
                lam * predicted.eq(targets_a.data).cpu().sum().float()
                + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float()
            )
            progress_bar(
                batch_idx,
                len(trainloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

    def test(testloader, namesave):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(
                    batch_idx,
                    len(testloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )
        # Save checkpoint.
        acc = 100.0 * correct / total
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, namesave)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=10, shuffle=False, num_workers=1
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=mini_batch_size, shuffle=True, num_workers=1
    )

    net = EfficientNet.from_pretrained("efficientnet-b4", num_classes=100)
    net = net.to(device)
    namesave = "./checkpoint/ckpt"
    optimizer = optim.SGD(
        net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
    )
    lr_sc = lr_scheduler.StepLR(optimizer, step_size=args.nsc, gamma=args.gamma)
    for epoch in range(0, args.ne):
        train(epoch, trainloader)
        lr_sc.step()

    print("Test accuracy : ")
    test(testloader, namesave)


if __name__ == "__main__":
    main()
