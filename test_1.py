import torch
import torch_geometric as tg
from torch_geometric.loader import DataLoader


torch.manual_seed(0)
#dataset = tg.datasets.Planetoid(root='data/Planetoid', name='Cora', transform=tg.transforms.NormalizeFeatures())
dataset = tg.datasets.WikiCS(root='data/wikics', transform=tg.transforms.NormalizeFeatures())
#dataset = tg.datasets.Planetoid(root='data/Planetoid', name='pubmed', transform=tg.transforms.NormalizeFeatures())
data = dataset[0]
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = tg.nn.GCN(-1,768, 5, 768, 768)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      pred = out.argmax(dim=1)
      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
      return test_acc


for epoch in range(25):
    loss = train()
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')