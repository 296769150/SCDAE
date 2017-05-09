# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

BatchSize = 64
# SIZE = 28
CUDA = True
EPOCHS = 20

def normalization(input):
    return torch.div(torch.add(input,-torch.mean(input)),torch.std(input))

def whiting(input):
    p = torch.mm(input,torch.t(input))
    U, S, V = torch.svd(p)
    T = torch.pow(S,-0.5)
    W = torch.mm(U,torch.diag(T))
    W = torch.mm(W,torch.t(V))
    return torch.mm(W,input)

def preprocess(input):
    output = normalization(input)
    return whiting(output)

def patch_select(input, k):
    """从batch中提取DAE的训练patch

    输入数据的维度为 Batch×Chan×Size×Size
    输出数据的维度为（Batch×（Size-k+1）×（Size-k+1））×（Chan×k×k）
    """
    Batch = input.size()[0]
    Chan = input.size()[1]
    Size = input.size()[2]
    #print Batch,Chan,Size
    #初始化
    output = torch.zeros(Batch*Chan*(Size-k+1)*(Size-k+1),k*k)

    for i in range(0, Batch):
        for j in range(0, Chan):
            for m in range(0, Size-k+1):
                for n in range(0, Size-k+1):
                    patch = torch.index_select(input[i][j],0,torch.LongTensor(range(m,m+k)))
                    patch = torch.index_select(patch,1,torch.LongTensor(range(n,n+k)))

                    if torch.equal(patch,torch.zeros(k,k)):
                        continue
                    else:
                        preprocess(patch)
                    patch.resize_(1,k*k)
                    output[((i*Chan+j)*(Size-k+1)+m)*(Size-k+1)+n] = patch
    #print "1",output.size()
    output.resize_(Batch*(Size-k+1)*(Size-k+1),Chan*k*k)
    #print "2", output.size()
    sq = []
    for i in range(0, output.size()[0]):
        if torch.equal(output[i], torch.zeros(k*k)):
            continue
        else:
            sq.append(i)
    indices = torch.LongTensor(sq)
    output = torch.index_select(output, 0, indices)

    return output

############    预处理    #####################################
class CNN1(nn.Module):
    def __init__(self,k1=None,k2=None):
        super(CNN1, self).__init__()

        self.cov1 = nn.Conv2d(1,1000,11,bias=False)
        self.pool1 = nn.MaxPool2d(6)
        self.cov2 = nn.Conv2d(1000,1500,2,bias=False)
        self.pool2 = nn.MaxPool2d(1)
        if k1 is not None:
            self.cov1.weight.data.copy_(k1)
        self.cov1.weight.requires_grad = False
        if k2 is not None:
            self.cov2.weight.data.copy_(k2)
        self.cov2.weight.requires_grad = False

        self.fc1 = nn.Linear(1500 * 2 * 2, 400)  # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.cov1(x)
        x = self.relu(self.pool1(x))
        z = self.cov2(x)
        z = self.relu(self.pool2(z))
        f = z.view(-1,1500*2*2)
        f = self.relu(self.fc1(f))
        f = F.dropout(f,training=self.training)
        f = self.relu(self.fc2(f))
        f = F.dropout(f, training=self.training)
        f = self.fc3(f)
        return F.log_softmax(f),x



def get_w(k,c,output_size,train_loader,sparsityParam=0.05,epochs=10,layer=True,k1=None):
    class DAE(nn.Module):
        def __init__(self):
            super(DAE, self).__init__()
            if layer:
                self.fc1 = nn.Linear(k*k*c,output_size)
                self.fc2 = nn.Linear(output_size,k*k*c)
            else:
                self.fc1 = nn.Linear(k * k * c, output_size)
                self.fc2 = nn.Linear(output_size, k * k * c)
            self.drop = nn.Dropout(p=0.5)
            self.sparse = nn.Dropout(p=sparsityParam)

            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def encode(self, x):
            x = self.drop(x)
            x = self.relu(self.fc1(x))
            return self.sparse(x)

        def decode(self, z):
            return self.relu(self.fc2(z))

        def forward(self, x):
            z = self.encode(x)
            return self.decode(z), z

    model = DAE()
    if CUDA:
        model.cuda()

    def loss_function(recon_x, x, z):
        BCE = torch.dist(recon_x, x)
        KLD_element = torch.mean(z)
        KLD = sparsityParam*torch.log(sparsityParam/KLD_element)\
              +(1-sparsityParam)*torch.log((1-sparsityParam)/(1-KLD_element))
        return BCE + KLD

    optimizer = optim.Adam(model.parameters())

    if not layer:
        CNN_model = CNN1(k1)
        if CUDA:
            CNN_model.cuda()



    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if layer:
                AE_train = patch_select(data, k)
                AE_train = Variable(AE_train)
            #flag = True 表示第1层
            else:
                data = Variable(data)
                if CUDA:
                    data = data.cuda()
                a, AE_train = CNN_model(data)  #64*1000*3*3
                AE_train = AE_train.cpu()
                AE_train = patch_select(AE_train.data, 2)
                AE_train = Variable(AE_train) #64*9000
            if CUDA:
                AE_train = AE_train.cuda()
            optimizer.zero_grad()
            recon_batch, z = model(AE_train)
            loss = loss_function(recon_batch, AE_train, z)

            loss.backward()
            train_loss += loss.data[0]

            optimizer.step()
            #just for debug
            # if batch_idx * len(data) >= 100:
            #     break

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0]/len(data) ))

    for epoch in range(1, epochs + 1):
        train(epoch)

    return model.fc1.weight

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=BatchSize, shuffle=True, num_workers=1, pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=BatchSize, shuffle=True, num_workers=1, pin_memory=True)


kernal1 = get_w(11,1,1000,train_loader,sparsityParam=0.05,epochs=10,layer=True)

kernal1 = kernal1.resize(1000,1,11,11)
kernal1 = kernal1.data
print "get kernal1"
kernal2 = get_w(2,1000,1500,train_loader,sparsityParam=0.05,epochs=10,layer=False,k1=kernal1)
print "get kernal2"
kernal2 = kernal2.resize(1500,1000,2,2)
kernal2 = kernal2.data


CNN_model1 = CNN1(kernal1,kernal2)
if CUDA:
    CNN_model1.cuda()

D_parameters = [
    {'params': CNN_model1.fc1.parameters()},
    {'params': CNN_model1.fc2.parameters()},
    {'params': CNN_model1.fc3.parameters()}
] # define a part of parameters in model

optimizer = optim.Adam(D_parameters)

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):

        if CUDA:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, useless = CNN_model1(data)  # 64*1000*3*3
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    CNN_model1.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if CUDA:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output, useless = CNN_model1(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test(epoch)
