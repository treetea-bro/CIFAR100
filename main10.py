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
from pathlib import Path


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
    parser.add_argument(
        "--model", default="efficientnet-b7", type=str, help="model name"
    )
    parser.add_argument(
        "--checkpoint_save_directory",
        default="./checkpoint",
        type=str,
        help="checkpoint_save_directory",
    )
    parser.add_argument(
        "--checkpoint_threshold",
        default=90.0,
        type=float,
        help="checkpoint_threshold",
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

    # 데이터로더에서 사용할 배치사이즈.
    # batch는 입력으로 받은 batch 값때마다 역전파를 진행하겠다는 의미이고,
    # batch_split은 입력으로 받은 batch_split 간격까지 gradient accumulation 하겠다는 의미.
    # 자세한건 아래 코드에서..
    mini_batch_size = args.batch // args.batch_split

    #  각 이미지에 적용할 변형기법을 파이프라인으로 정의
    #  이미지는 각 함수를 통과하여 변형됨
    transform_train = transforms.Compose(
        [
            # 보간 기법으로 BILINEAR를 사용하여 160x160사이즈로 이미지를 변환시킨다.
            # 보간 기법을 적용하면 이미지 해상도가 낮아져서 성능이 안좋을줄 알았지만,
            # efficientnet 특성상 이미지 크기가 커야 학습 성능이 더 잘나온다.
            transforms.Resize(160),
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
            transforms.Resize(200),
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
        """입력으로 받은 x에 해당하는 이미지들을 mixup해서 데이터 증강
        효과를 볼 수 있게하는 함수"""
        # count가 0이 되는 주기마다 alpha값이 존재할 시 mixup을 진행한다.
        # count의 주기는 args.batch_split이 결정하며,
        # 현재 코드에선 args.batch_split이 1이고, 매 번 mixup을 진행한다.
        if count == 0:
            if alpha > 0:
                # 현재 args 값에 alpha는 0.1로 설정되어있어서
                # lam의 값은 0이나 1에 가까운 값이 난수로 생성되게 된다.
                lam = np.random.beta(alpha, alpha)
            else:
                # lambda가 1인 경우는 mixup을 하지 않는다.
                # 현재 이 코드에선 alpha가 존재하기 때문에 매번 mixup한다.
                lam = 1.0

        # 매 미니배치(32)마다 이 함수가 호출되는데, 그때마다
        # 순서가 0-31의 인덱스 이미지와 0-31범위에 해당하긴하지만
        # 순서가 랜덤으로 생성된 이미지 ex: [1,4,6,8,0,2 ...]
        # 를 mixup 하기위해 batch_size(32)를 인자로 랜덤순열을 생성한다.
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)

        # 위에서 난수로 구한 lam값과 기존 배치 0-31 인덱스와
        # 랜덤순열로 구한 인덱스를 가지고 이미지들을 mix한다.
        mixed_x = lam * x + (1 - lam) * x[index, :]

        # y_a는 0-31인덱스의 정답, y_b는 mixup에 사용된 이미지들의 정답
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        """loss function(criterion)과 모델의 예측값, mixup 라벨 2개를 받아서
        각각 라벨당 오차를 계산하고 더한 뒤 리턴한다.
        lam값이 1일 경우 y_a에 대한 loss값만 리턴한다."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    # Training
    def train(epoch, trainloader) -> tuple[Path, float]:
        """모델 학습을 담당하는 함수"""
        # 현재 Epoch가 몇번째인지 출력한다.
        print("\nEpoch: %d" % epoch)
        # 모델을 학습모드로 설정한다.
        # 학습모드에서는 gradient 계산 및 dropoutd이 활성화된다.
        net.train()

        # 현재 epoch에서의 가장 높은 정확도를 구한다.
        # checkpoint 저장 시에 활용된다.
        max_accuracy = 0.0

        # 아래 4개의 지역변수들은 progress_bar 출력을 위해 사용된다.
        train_loss = 0
        correct = 0
        total = 0
        count = 0

        # lam 지역변수는 mixup에서 모델의 예측값이 정답인지 확인할 때 쓴다.
        # mixup_data 함수를 거쳐나오면 lam값이 mutable하게 변하기때문에 항상 1.0이지는 않다.
        lam = 1.0

        # 학습 시작 전에 optimizer의 gradient들을 초기화 해준다.
        optimizer.zero_grad()

        saving_ckpt_path = Path("")
        # trainloader에서 설정한 미니배치 단위의 묶음으로 이미지와 정답라벨을 순회하며 가져온다.
        # batch_idx는 progress_bar 출력을 위해 사용된다.
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # count가 batch_split과 같아지면 loss function에서 누적계산 해놓은 값을 업데이트한다.
            # batch_split과 같아지기위해 count는 매 루프마다 1씩 증가하는 코드가 아래 존재한다.
            if count == args.batch_split:
                # gradient를 업데이트 한다.
                optimizer.step()
                # 새 업데이트를 위해 이전 업데이트에 사용된 값들을 초기화한다.
                optimizer.zero_grad()
                # count를 0으로 맞춰주어서 일정 주기마다 이 if문 안으로 들어올 수 있도록 한다.
                # 예를들어 batch_split이 2라면 짝수 주기로 gradient 업데이트를하고,
                # 1이면 매 루프마다 업데이트한다.
                count = 0
            # batch_split 주기마다 gradient 업데이트 하기위해 count를 1씩 증가해준다.
            count += 1
            # GPU를 사용할 수 있다면 tensor 객체를 GPU에 적재하고, 그렇지 않다면 CPU에 적재한다.
            inputs, targets = inputs.to(device), targets.to(device)

            # inputs: mixup된 이미지
            # targets_a: mixup 중 alpha에 해당하는 이미지 정답라벨
            # targets_b: mixup 중 beta에 해당하는 이미지 정답라벨
            # lam: 이미지 섞인 비율 값
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, targets, args.alpha, lam, count
            )
            # efficientnet-b7에 mixup 이미지를 넣어서 predict한다.
            outputs = net(inputs)
            # mixup 이미지의 loss를 계산하는 함수를 이용하여 loss를 계산한다.
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            # batch_split이 2 이상이면 gradient accumulation을 진행하니,
            # loss를 일부분씩만 각각 미니배치에서 취해서 누적계산을 진행한다.
            loss = loss / args.batch_split
            # 역전파 진행
            loss.backward()

            # 매 루프마다 train_loss를 누적증가 시킨다.
            train_loss += loss.item()
            # 모델이 예측한 확률 중 가장 높은 라벨만 골라서 추출한다.
            # 1차원 벡터에 32개의 값들이 추출된다. (batch_size가 32여서)
            _, predicted = outputs.max(1)
            # 현 배치의 개수만큼 total에 누적합 한다.
            total += targets.size(0)
            # mixup 이미지를 대상으로 했기때문에 정답을 계산하는 방식도 다르다.
            # target a와 b를 모델이 맞췄다면 lam을 활용해서 비율 조정해서 맞았다고 처리한다.
            correct += (
                lam * predicted.eq(targets_a.data).cpu().sum().float()
                + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float()
            )
            # accuracy : 전체데이터/  맞은 비율 * 100.0
            acc = 100.0 * correct / total

            # 모델의 학습진행상태를 출력한다.
            progress_bar(
                batch_idx,
                len(trainloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    acc,
                    correct,
                    total,
                ),
            )

            # 모델이 예측한 정확도가 checkpoint threshold 이상이고, 이전 max를 넘었다면
            # checkpoint를 저장한다.
            if acc > args.checkpoint_threshold and max_accuracy < acc.item():
                max_accuracy = round(acc.item(), 2)
                saving_ckpt_path = Path(args.checkpoint_save_directory) / Path(
                    f"{args.model}_{max_accuracy}.pt"
                )
                print(f"Saving model : {saving_ckpt_path}")
                state = {
                    "net": net.state_dict(),
                    "acc": acc,
                }
                # checkpoint 폴더가 존재하지 않으면 만들어준다.
                if not os.path.isdir(args.checkpoint_save_directory):
                    os.mkdir(args.checkpoint_save_directory)
                torch.save(state, saving_ckpt_path)

        return saving_ckpt_path, max_accuracy

    def test(testloader, saving_ckpt_path: Path) -> None:
        """모델 평가를 담당하는 함수"""
        # checkpoint 가 존재하지 않는다면 테스트를 진행하지 않는 guard
        if not (saving_ckpt_path.exists() and saving_ckpt_path.is_file()):
            print("입력하신 체크포인트 파일이 경로에 존재하지 않습니다.")
            return

        # 모델을 평가모드로 전환해서 dropout 및 gredient 변동이 일어나지 않게한다.
        net.eval()
        print("Loading checkpoint..")
        # 체크포인트를 load한다.
        checkpoint = torch.load(saving_ckpt_path)
        # load한 체크포인트를 모델에 적용한다.
        net.load_state_dict(checkpoint["net"])

        test_loss = 0
        correct = 0
        total = 0
        print(f"{'-' * 10} 모델 테스트 시작 {'-' * 10}")
        # gradient를 계산하지 않는 상태에서 모델의 정확도를 측정한다.
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

    # 데이터로더를 이용해서 배치사이즈 크기별로 iterate 할 수 있도록 한다.
    # 테스트는 shuffle을 하든 안하든 상관없기 때문에 성능상의 이유로 False이고,
    # 트레인은 shuffle을 해야 매 에포크 및 미니배치마다 다양한 조합의 이미지들이
    # 배치 정규화 및 mixup 되기때문에 하면 일반화 및 모델 성능에 좋다.
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=10, shuffle=False, num_workers=1
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=mini_batch_size, shuffle=True, num_workers=1
    )

    # 모델은 EfficientNet을 이용한다.
    net = EfficientNet.from_pretrained(args.model, num_classes=100)
    net = net.to(device)
    optimizer = optim.SGD(
        net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
    )
    # StepLR을 이용해서 에포크가 진행될수록 lr를 점점 줄이도록 한다.
    lr_sc = lr_scheduler.StepLR(optimizer, step_size=args.nsc, gamma=args.gamma)

    # train
    highest_model_path = Path("")
    max_accuracy = 0.0
    for epoch in range(0, args.ne):
        model_path, accuracy = train(epoch, trainloader)
        # 전체 에포크 중에서 가장 높은 정확도를 가진 모델을 구한다.
        # 그 모델로 나중에 test함수에 넣어서 평가하기 위한 용도이다.
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            highest_model_path = model_path
        # learning rate를 줄인다.
        lr_sc.step()

    if str(highest_model_path) == "":
        print(
            "checkpoint_threshold가 너무 높아 모델이 저장되지 않았습니다.\nthreshhold arg를 낮춰주세요."
        )
        return

    test(testloader, highest_model_path)


if __name__ == "__main__":
    main()
