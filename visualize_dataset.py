import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dissnn.dataset import NonlinearOscillatorDataset


file = 'data/oscillator2_11node_sic/train.npz'
dataset = NonlinearOscillatorDataset(file=file)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, (x0, x) in enumerate(dataloader):
    plt.plot(x[0, :, 0, 0].detach().numpy(), x[0, :, 0, 1].detach().numpy())

plt.show()
