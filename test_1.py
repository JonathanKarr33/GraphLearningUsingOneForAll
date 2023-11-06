import torch
import torch_geometric as tg
from torch_geometric.loader import DataLoader

device = torch.device("cuda:1")


torch.manual_seed(0)
#dataset = tg.datasets.Planetoid(root='data/Planetoid', name='Cora', transform=tg.transforms.NormalizeFeatures())
dataset = tg.datasets.WikiCS(root='data/wikics', transform=tg.transforms.NormalizeFeatures(),is_undirected=True)
#dataset = tg.datasets.Planetoid(root='data/Planetoid', name='pubmed', transform=tg.transforms.NormalizeFeatures())
data = dataset[0]
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = tg.nn.GCN(dataset.num_features,768, 5,dataset.num_classes,jk='last').to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      for batch in loader:
        #print(batch.train_mask.shape)
        optimizer.zero_grad()
        out = model(batch.x.to(device), batch.edge_index.to(device))
        loss = criterion(out[batch.train_mask[:,:10]], torch.nn.functional.one_hot(batch.y).float()[batch.train_mask[:,:10]].to(device))
        loss.backward()
        optimizer.step()
      return loss
# def train():
#       model.train()
#       for batch in loader:
#         #print(batch.train_mask.shape)
#         optimizer.zero_grad()
#         out = model(batch.x.to(device), batch.edge_index.to(device))
#         loss = criterion(out[batch.train_mask], batch.y[batch.train_mask].to(device))#torch.nn.functional.one_hot(batch.y).float()[batch.train_mask[:,10:]].to(device))
#         loss.backward()
#         optimizer.step()
#       return loss

def test():
      model.eval()
      out = model(data.x.to(device), data.edge_index.to(device))
      pred = out.argmax(dim=1).to('cpu')
      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
      return test_acc


for epoch in range(500):
    loss = train()
    print("epoch {}: ".format(epoch+1),loss)
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')