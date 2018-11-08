import xarray as xr
from uwnet.model import ApparentSource
from uwnet.datasets import _ds_slice_to_torch, XarrayPoints
from toolz import curry
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from uwnet.loss import dictloss
from ignite.engine import Engine, Events, create_supervised_trainer
import matplotlib.pyplot as plt
plt.ion()


class TupleDataset(Dataset):
    def __init__(self, datasets):
        super(TupleDataset, self).__init__()
        self.datasets = datasets

    def  __len__(self):
        lens = set(len(dataset) for dataset in self.datasets)
        same_length = len(lens) == 1
        if not same_length:
            raise ValueError("The datasets do not have the same length")
        return lens.pop()

    def __getitem__(self, i):
        return list(dataset[i] for dataset in self.datasets)


def get_data_loaders(train_batch_size, val_batch_size):
    return iter(range(10))


def _ds_slice_to_torch(x):
    return {key: torch.tensor(val.values).view(-1, 1, 1) for key, val in x.items()}


def get_mean_sig(data):
    avg_dims = ['x', 'y', 'time']
    mean = data.mean(avg_dims)
    sig = data.std(avg_dims)
    sig = sig.where(sig > 1e-6, 1e-6)

    mean = _ds_slice_to_torch(mean)
    sig = _ds_slice_to_torch(sig)

    return mean, sig


def get_model(data):
    vertical_grid_size = 34
    data = data.isel(z=slice(0, vertical_grid_size))
    input_fields = (('QT', vertical_grid_size),
                    ('SLI', vertical_grid_size), ('SST', 1), ('SOLIN', 1))
    output_fields = (('QT', vertical_grid_size), ('SLI', vertical_grid_size))
    mean, sig = get_mean_sig(data)
    return ApparentSource(mean, sig, inputs=input_fields,
                          outputs=output_fields)


def get_data_loader(data, batch_size=128):
    fields = ['QT', 'SLI', 'FQT', 'FSLI', 'SST', 'SOLIN']
    progs = ['QT', 'SLI']

    data = data.load()

    inputs = data[fields].isel(time=slice(0, -1))
    output = data[progs].shift(time=-1).dropna('time')

    dataset = TupleDataset([
        XarrayPoints(inputs),
        XarrayPoints(output)
    ])

    return DataLoader(dataset, batch_size=batch_size)


data_path = "./data/processed/training.nc"
dt = .125

data = xr.open_dataset(data_path).isel(step=0, y=slice(32, 33)).load()
model = get_model(data)
train_loader = get_data_loader(data)
optimizer = torch.optim.Adam(model.parameters(), lr=.01)
loss_fn = dictloss(nn.MSELoss(), ['QT', 'SLI'])



def _add_null_dims(x):
    def _(x):
        return x.unsqueeze(-1).unsqueeze(-1)
    return {key: _(val) for key, val in x.items()}


def _remove_null_dims(x):
    def _(x):
        return x.squeeze(-1).squeeze(-1)
    return {key: _(val) for key, val in x.items()}

class EulerStep(nn.Module):
    def __init__(self, model, keys, dt):
        super(EulerStep, self).__init__()

        self.model = model
        self.keys = keys
        self.dt = dt

    def forward(self, x):
        x = _add_null_dims(x)
        prediction = {}
        sources = self.model(x)
        for key in self.keys:
            state = x[key]
            forcing_key = 'F' + key
            prediction[key] = state + dt * sources[key] + dt * 86400 * x[forcing_key]
        return _remove_null_dims(prediction)


stepper = EulerStep(model, ['QT', 'SLI'], dt)

def train_and_store_loss(engine, batch):
    x, y = batch
    x = _add_null_dims(x)
    y = _add_null_dims(y)

    predictions = stepper(x)
    loss = loss_fn(predictions, y)/dt
    loss.backward()
    optimizer.step()
    return loss.item()



def test_train_loader():
    train_loader = get_data_loader(data)
    x, y = next(iter(train_loader))
    _add_null_dims(x)

    train_and_store_loss(None, (x, y))


trainer = create_supervised_trainer(stepper, optimizer, loss_fn)


desc = "ITERATION - loss: {:.2f}"
pbar = tqdm(
    initial=0, leave=False, total=len(train_loader),
    desc=desc.format(0)
)

log_interval = 1
@trainer.on(Events.ITERATION_COMPLETED)
def log_training_loss(engine):
    iter = (engine.state.iteration - 1) % len(train_loader) + 1

    if iter % log_interval == 0:
        pbar.desc = desc.format(engine.state.output)
        pbar.update(log_interval)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    pbar.refresh()
    # evaluator.run(train_loader)
    # metrics = evaluator.state.metrics
    # avg_accuracy = metrics['accuracy']
    # avg_nll = metrics['nll']
    output = model.call_with_xr(data.isel(x=slice(0,1)))
    plt.figure()
    output.QT.plot(x='time')
    plt.xlim([100, 120])
    plt.savefig(f"{engine.state.epoch}.png")
    plt.close()

    tqdm.write(
        "Training Results - Epoch: {} "
        .format(engine.state.epoch)
    )
    pbar.n = pbar.last_print_n = 0




trainer.run(train_loader, max_epochs=4)
pbar.close()

# engine.run(Dumb())
# device = 'cpu'

# if torch.cuda.is_available():
#     device = 'cuda'

# optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
# trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
# evaluator = create_supervised_evaluator(model,
#                                         metrics={'accuracy': Accuracy(),
#                                                     'nll': Loss(F.nll_loss)},
#                                         device=device)

# desc = "ITERATION - loss: {:.2f}"
# pbar = tqdm(
#     initial=0, leave=False, total=len(train_loader),
#     desc=desc.format(0)
