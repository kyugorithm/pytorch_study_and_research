import sys # system version check
import sklearn.datasets # mnist 데이터셋 로드
import torch # torch 관련 메서드 사용
import matplotlib.pyplot as plt # 로드 데이터 유효성 판단을 위해 plotting

print(f'python  : {sys.version}') ## system 버전 체크
print(f'pytorch : {torch.__version__}') ## torch 버전 체크

mnist = sklearn.datasets.fetch_openml(name='mnist_784', data_home='mnist_784') # openml 데이터 repository의 mnist_784 데이터 로드

x_train = torch.tensor(mnist.data[:60000], dtype=torch.float) / 255 # 70k개의 이미지 중 60k의 이미지를 사용하여 학습 /  255로 나누어 normalize
y_train = torch.tensor([int(x) for x in mnist.target[:60000]], dtype=torch.float) # target은 object형 type이기 때문에 format을 맞춰주기 위해 int로 변환하여 torch.tensor에 할당
x_test  = torch.tensor(mnist.data[60000:], dtype=torch.float) / 255 # 70k개의 이미지 중 10k의 이미지를 사용하여 테스트 /  255로 나누어 normalize
y_test  = torch.tensor([int(x) for x in mnist.target[60000:]], dtype=torch.float)

fig, axes = plt.subplots(2, 4, constrained_layout = True) # 새로운 figure 선언,  2행/4열 : constrained_layout : subplot간 간격을 최적으로 설정해줌
for i, ax in enumerate(axes.flat):
    ax.imshow(1 - x_train[i].reshape((28,28)), cmap = 'gray', vmin = 0, vmax = 1)
    ax.set(title = f'{y_train[i]}')
    ax.set_axis_off()
#%%
def log_softmax(x):
    return x - x.exp().sum(dim=-1).log().unsqueeze(-1) # log e ^(x) == x 이므로 간소화된 log의 softmax 표현 사용

def model(x, weights, bias):
    return log_softmax(torch.mm(x, weights) + bias) # 단일 레이어의 모델이기 때문에 input x output의 connection을 가지도록 W 정의하고 사용(뒤에서)

def neg_likelihood(log_pred, y_true): # negative cross entropy 사용 : entropy revisit 필요!
    return -log_pred[torch.arange(y_true.size()[0]).long(), y_true.long()].mean()  # 입력 모든 값을 평균화하여 expectation calc.

def accuracy(log_pred, y_true): # pred 추정 class를 추출하여 맞는경우의 평균으로 정확도 계산
    y_pred = torch.argmax(log_pred, dim=1)
    return (y_pred == y_true).to(torch.float).mean()

def print_loss_accuracy(log_pred, y_true, loss_function): # Loss/Accuracy 추출 print시 f'' 자주 실수하므로 놓치지 말자.
    with torch.no_grad(): # 해당 연산을 수행할 때 grad 계산을 수행하지 않기 위해 no_grad로 감싼다. @torch.no_grad()도 유효함 : 뭐라고 표현하지?
        print(f"Loss: {neg_likelihood(log_pred, y_true):.6f}")
        print(f"Accuracy: {100 * accuracy(log_pred, y_true).item():.2f} %")

loss_function = neg_likelihood

# %%
batch_size = 100
learning_rate = 0.5
n_epochs = 50

weights = torch.randn(784, 10, requires_grad=True)
bias = torch.randn(10, requires_grad=True)

for epoch in range(n_epochs):
    # Batch 반복
    for i in range(x_train.size()[0] // batch_size):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_batch = x_train[start_index:end_index]
        y_batch_true = y_train[start_index:end_index]

        # Forward
        y_batch_log_pred = model(x_batch, weights, bias)
        loss = loss_function(y_batch_log_pred, y_batch_true)

        # Backward
        loss.backward()

        # Update
        
        weights.data -= learning_rate * weights.grad
        bias.data -= learning_rate * bias.grad

        # Zero the parameter gradients
        weights.grad = None
        bias.grad = None

    with torch.no_grad():
        y_test_log_pred = model(x_test, weights, bias)
    print(f"End of epoch {epoch + 1}")
    print_loss_accuracy(y_test_log_pred, y_test, loss_function)
    print("---")
