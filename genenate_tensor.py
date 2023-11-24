import torch

a = torch.load("a.pt")
b = torch.load("b.pt")
c = torch.load("c.pt")
d = torch.load("d.pt")

tensor = torch.cat((a,b,c,d), dim=0)
torch.save(tensor,'tensor.pt')
print(tensor.size())