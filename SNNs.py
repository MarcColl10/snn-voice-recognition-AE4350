import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from speech2spikes import S2SPreProcessor
from neurobench.datasets.MSWC_dataset import MSWC
from neurobench.processors.postprocessors import ChooseMaxCount
from neurobench.metrics.workload import (
    ActivationSparsity,
    SynapticOperations,
    ClassificationAccuracy,
)
from neurobench.metrics.static import (
    Footprint,
    ConnectionSparsity,
)
from neurobench.benchmarks import Benchmark
from neurobench.models import SNNTorchModel
import snntorch.functional as SF


class SNNModel(nn.Module):
    def __init__(self, beta=0.95):
        super(SNNModel, self).__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.fc1 = nn.Linear(20, 600)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad,threshold= 0.3, init_hidden=True)

        self.fc2 = nn.Linear(600, 100)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad,threshold= 0.5, init_hidden=True,output=True)


    def forward(self, x):
        if x.dim() == 2: 
            self.lif1.init_leaky() 
            self.lif2.init_leaky() 
            
            xt = self.fc1(x) 
            xt = self.lif1(xt) 

            xt = self.fc2(xt) 
            spk, mem = self.lif2(xt) 

            
            return spk, mem # matches NeuroBench expectations for benchmarking
            
        self.lif1.init_leaky()
        self.lif2.init_leaky()

        batch_size, time_steps, _ = x.shape
        outputs = []
        spk2_rec = []

        for t in range(time_steps):
            xt = x[:, t, :]
            xt = self.fc1(xt)
            xt = self.lif1(xt)

            xt = self.fc2(xt)
            spk, mem = self.lif2(xt)

            outputs.append(mem)
            spk2_rec.append(spk)
        
        out = torch.stack(outputs, dim=0)
        return  out


# Load the dataset
path = r"/home/Marc/Desktop/University TU Delft/Bio Inspired/"  # Update to your desired location
train_set = MSWC(path, procedure="training", subset="base", language="ca")
val_set = MSWC(path, procedure="validation", subset="base", language="ca")
test_set = MSWC(path, procedure="testing", subset="base",language="ca")

# Create DataLoaders
train_set_loader = DataLoader(train_set, batch_size=256, shuffle=True,
    collate_fn=lambda batch: (
        torch.stack([item[0] for item in batch]),  # Audio
        torch.tensor([item[1] for item in batch], dtype=torch.long)  # Labels
    )
)

val_set_loader = DataLoader(val_set, batch_size=256, shuffle=False,
    collate_fn=lambda batch: (
        torch.stack([item[0] for item in batch]),  # Audio
        torch.tensor([item[1] for item in batch], dtype=torch.long)  # Labels
        )
)

test_set_loader = DataLoader(test_set, batch_size=256, shuffle=False,
    collate_fn=lambda batch: (
        torch.stack([item[0] for item in batch]),  # Audio
        torch.tensor([item[1] for item in batch], dtype=torch.long)  # Labels
    )
)


def train(train_loader, num_epochs=600, lr=1e-3):
    s2s = S2SPreProcessor()
    s2s.configure(threshold=0.2)
    model = SNNModel()
    model.train()
    criterion = SF.ce_max_membrane_loss() 
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    print("Training SNN model...")
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        epoch_loss=[]
        for batch in tqdm(train_loader, desc="Training Batches"):
                
            audio, labels = batch
            spike_trains, _ = s2s((audio, labels)) # Use S2S to convert to spikes

            # Forward pass
            y = model(spike_trains)
            loss = criterion(y, labels)
            epoch_loss.append(loss.item())


            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdm.write(f"Batch Loss: {loss.item():.4f} | Epoch: {epoch + 1}/{num_epochs}")

        
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < 0.1:
            tqdm.write("Early stopping: Loss below threshold.")
            break

    tqdm.write("Training complete.")
    model = model.eval()
    return model

Train = True  # Set to True to train the model, False to skip training
if Train == True:
    trained_model = train(train_set_loader)
    torch.save(trained_model.state_dict(), "snn_model.pth")
    trained_model = SNNTorchModel(trained_model)  # wrap for benchmarking
else:
    model = SNNModel()
    model.load_state_dict(torch.load("snn_model.pth", map_location="cpu"))
    trained_model = SNNTorchModel(model)  # wrap for benchmarking



# Benchmarking setup
preprocessors = [S2SPreProcessor()]
postprocessors = [ChooseMaxCount()]
static_metrics = [Footprint, ConnectionSparsity]
workload_metrics = [ClassificationAccuracy, ActivationSparsity, SynapticOperations]

with torch.no_grad():
    print('Validation Results')
    benchmark_val = Benchmark(
        model=trained_model,
        dataloader=val_set_loader,
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        metric_list=[static_metrics, workload_metrics]
    )   
    results_val = benchmark_val.run()
    print(results_val)


    print('Test Results')
    benchmark_test = Benchmark(
        model=trained_model,
        dataloader=test_set_loader,
        preprocessors=preprocessors,
        postprocessors=postprocessors,
        metric_list=[static_metrics, workload_metrics]
    )
    results_test = benchmark_test.run()
    print(results_test)
