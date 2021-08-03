## stack()과 cat()의 차이
torch에는 tensor를 합치는 기능을하는 두개의 method가 있다.
stack과 cat이 그것인데, stack은 말그대로 쌓아올리는것을 생각할 수 있다.  
블럭을 쌓아올리는 상황을 생각해 보면 하나의 블럭은 특정 차원을 가진 tensor로 표현할 수 있고 쌓아나가는 것은
새로운 방향으로 만들어지는것으로 표현할 수 있다.  
cat은 단순히 지정한 기존의 축 방향으로 tensor를 추가해 나가는것을 의미한다. 기존의 concatenate과 동일하게 이해하면 된다.


![image](https://user-images.githubusercontent.com/40943064/128004245-56d292d8-02ec-4ace-b914-9866ea9d01c8.png)
### 1. stack()
```{.python}
import torch
a   = torch.rand(100, 512)
b   = torch.rand(100, 512)
rst = torch.stack([a,b], dim=0)
print(rst.shape)
```
```{.python}
torch.Size([2, 100, 512])
```

### 2. cat()
```{.python}
import torch
a   = torch.rand(100, 512)
b   = torch.rand(100, 512)
rst = torch.cat([a,b], dim=0)
print(rst.shape)
```
```{.python}
torch.Size([200, 512])
```
