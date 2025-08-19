# implementing LeNet5 in tinygrad

from tinygrad import Tensor, TinyJit, nn, GlobalCounters
import tinygrad.nn as nn
from tinygrad.nn.datasets import mnist
from tqdm import trange

# (MNIST is 28x28)

class LeNet5:
    def __init__(self):
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=4)
        
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
        
    def __call__(self, x: Tensor)->Tensor:
        x = self.conv1(x).relu().avg_pool2d(kernel_size=2, stride=2)
        x = self.conv2(x).relu().avg_pool2d(kernel_size=2, stride=2)
        x = self.conv3(x).relu()
        x = x.flatten(1)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

if __name__=="__main__":
    X_train, Y_train, X_test, Y_test = mnist()
    
    model = LeNet5()
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=1e-3)
    
    BS = 128
    
    @TinyJit
    @Tensor.train()
    def train_step():
        opt.zero_grad()
        samples = Tensor.randint(BS, high=X_train.shape[0])
        x, y, = X_train[samples], Y_train[samples]
        out = model(x)
        loss = out.sparse_categorical_crossentropy(y)
        loss.backward()
        opt.step()
        return loss
    
    @TinyJit
    def get_test_acc():
        return (model(X_test).argmax(axis=1) == Y_test).mean() * 100

    epochs = 1000
    for i in range(epochs):
        GlobalCounters.reset()
        loss = train_step()
        if i % 5 == 4:
            acc = get_test_acc().item()
            print(f"epoch {i} | loss {loss.item():.4f} | test_acc {acc:.2f}%")

            
