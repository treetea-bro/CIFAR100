# import torch
#
#
# class cutout:
# """랜덤한 위치와 크기의 정사각형으로 이미지의 일부분을 검은색으로 변환하여
# 학습 시에 모델이 좀 더 보편적으로 이미지를 학습할 수 있게 도와주는 기법(즉 일반화를 뜻함)"""
#
#     def __init__(self, size=16, p=0.5):
#         self.size = size  # 정사각형 한변의 길이
#         self.half_size = size // 2
#         self.p = p  # cutout을 진행할 확률
#
#     def __call__(self, image):
#         # 0~1 사이의 랜덤하게 생성된 값이 self.p 이상이면 cutout 중지
#         if torch.rand([1]).item() > self.p:
#             return image
#
#         # 이미지의 좌측과 위측의 좌표를 랜덤하게 구하고,
#         # 미리 정의된 한변의 길이를 더해서 우측과 아래측의
#         # 좌표를 구한다.
#         left = torch.randint(
#             -self.half_size, image.size(1) - self.half_size, [1]
#         ).item()
#         top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
#         right = min(image.size(1), left + self.size)
#         bottom = min(image.size(2), top + self.size)
#
#         # 구해진 좌표를 가지고 슬라이싱을 이용해 이미지의 일부분을 검은색(0)으로 만든다.
#         image[:, max(0, left) : right, max(0, top) : bottom] = 0
#         return image

import torch


class Cutout:
    """랜덤한 위치와 크기의 정사각형으로 이미지의 일부분을 검은색으로 변환하여
    학습 시에 모델이 좀 더 보편적으로 이미지를 학습할 수 있게 도와주는 기법(즉 일반화를 뜻함)"""

    def __init__(self, min_side=30, max_side=60, p=0.5):
        self.max_side = max_side  # 정사각형 한변의 최대 길이
        self.min_side = min_side  # 정사각형 한변의 최소 길이
        self.p = p  # cutout을 진행할 확률

    def __call__(self, image):
        # 0~1 사이의 랜덤하게 생성된 값이 self.p 이상이면 cutout 중지
        if torch.rand([1]).item() > self.p:
            return image

        # 정사각형 한변의 길이를 min_side와 max_side 사이에서 랜덤하게 생성
        side = torch.randint(self.min_side, self.max_side + 1, [1]).item()

        # 이미지의 좌측과 위측의 좌표를 랜덤하게 구하고 (0 ~ image.size(1|2) - side),
        # 랜덤으로 구한 정사각형 한변의 길이를 더해서 우측과 아래측의 좌표를 구한다.
        left = torch.randint(0, image.size(1) - side, [1]).item()
        top = torch.randint(0, image.size(2) - side, [1]).item()
        right = left + side
        bottom = top + side

        # 구해진 좌표를 가지고 슬라이싱을 이용해 이미지의 일부분을 검은색(모든채널을 0)으로 만든다.
        image[:, left:right, top:bottom] = 0
        return image
