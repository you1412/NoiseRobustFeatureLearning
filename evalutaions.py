import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassNet(nn.Module):
    def __init__(self,device, z_dim):
        super(ClassNet, self).__init__()
        self.device=device
        self.fc1= nn.Linear(z_dim, 200)
        self.dropout=nn.Dropout(0.1)
        self.bn=nn.BatchNorm1d(200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self,x):
        h=F.relu(self.fc1(x))
        h=self.fc2(self.bn(self.dropout(h)))
        return torch.log_softmax(h,dim=-1)

    def predict(self,x):
        p=self.forward(x)
        pred=p.argmax(dim=-1)
        return pred


def LinerSVM(model,train,test,opt):
    from sklearn.svm import SVC
    model.eval()
    clf=SVC(kernel='linear')
    representation_list=[]
    label_list=[]
    with torch.no_grad():
        for x, y in train:
            z=model.encoder(x.to(opt['device']))
            representation_list.append(z.cpu().detach())
            label_list.append(y.numpy())
    clf.fit(torch.stack(representation_list).view(-1,opt['z_dim']).numpy(),np.asarray(label_list).reshape(-1))

    representation_list=[]
    label_list=[]
    with torch.no_grad():
        for x, y in test:
            z=model.encoder(x.to(opt['device']))
            representation_list.append(z.cpu().detach())
            label_list.append(y.numpy())
    accuracy=clf.score(torch.stack(representation_list).view(-1,opt['z_dim']).numpy(), np.asarray(label_list).reshape(-1))

    return accuracy