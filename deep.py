# deep.py
# PyTorch로 손글씨 숫자 인식 딥러닝!

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# =============================================
# ① 데이터 불러오기
# =============================================

# MNIST 데이터 자동 다운로드!
# transforms.ToTensor() = 이미지를 숫자로 변환
train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor()
)

# DataLoader = 데이터를 64개씩 묶어서 줘요
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=64, shuffle=False)

print(f"✅ 학습 데이터: {len(train_data)}장")
print(f"✅ 시험 데이터: {len(test_data)}장")

# =============================================
# ② 딥러닝 모델 만들기
# =============================================

# nn.Module = PyTorch 모델의 기본 클래스
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 층 정의
        self.layers = nn.Sequential(
            nn.Flatten(),           # 28x28 → 784
            nn.Linear(784, 128),    # 784 입력 → 128 뉴런
            nn.ReLU(),              # 음수는 0으로
            nn.Dropout(0.2),        # 20% 랜덤으로 끄기
            nn.Linear(128, 10)      # 128 → 10 (숫자 0~9)
        )

    def forward(self, x):
        return self.layers(x)

model = MyModel()
print(f"\n✅ 모델 생성 완료!")
print(model)

# =============================================
# ③ 학습 준비
# =============================================

# 오차 계산 함수
criterion = nn.CrossEntropyLoss()

# 학습 방법 (Adam = 제일 많이 쓰는 방법)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =============================================
# ④ 학습!
# =============================================

print("\n🔥 학습 시작!")

epochs = 5  # 전체 데이터 5번 반복

for epoch in range(epochs):
    model.train()  # 학습 모드
    total_loss = 0

    for images, labels in train_loader:
        # 예측
        outputs = model(images)

        # 오차 계산
        loss = criterion(outputs, labels)

        # 역전파 (오차를 줄이는 방향으로 학습)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

# =============================================
# ⑤ 정확도 평가
# =============================================

model.eval()  # 평가 모드
correct = 0
total = 0

with torch.no_grad():  # 평가할 땐 학습 안 해요
    for images, labels in test_loader:
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total * 100
print(f"\n🎯 최종 정확도: {accuracy:.1f}%")

# =============================================
# ⑥ 직접 예측해보기
# =============================================

# 테스트 이미지 1장 예측
sample_image, sample_label = test_data[0]
output = model(sample_image.unsqueeze(0))
predicted = torch.argmax(output).item()

print(f"\n예측 결과: {predicted}")
print(f"실제 정답: {sample_label}")
