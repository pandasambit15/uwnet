import argparse
import logging
import os

import yaml

import torch
import torchnet as tnt
from torch.utils.data import DataLoader
from toolz import curry
from uwnet import model
from uwnet.data import get_dataset
from uwnet.utils import get_batch_size, select_time


def mse(x, y, layer_mass):
    x = x.float()
    y = y.float()
    layer_mass = layer_mass.float()
    w = layer_mass / layer_mass.mean()

    if x.dim() == 2:
        x = x[..., None]

    if x.size(-1) == w.size(-1):
        return torch.mean(torch.pow(x - y, 2) * w)
    else:
        return torch.mean(torch.pow(x - y, 2))

@curry
def MVLoss(layer_mass, scale, x, y):
    """MSE loss

    Parameters
    ----------
    x : truth
    y : prediction
    """
    losses = {key: mse(x[key], y[key], layer_mass)/torch.tensor(scale[key]**2).float()
              for key in scale}
    return sum(losses.values())


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', '--restart', default=False)
    parser.add_argument('-lr', '--lr', default=.001, type=float)
    parser.add_argument('-n', '--n-epochs', default=10, type=int)
    parser.add_argument('-o', '--output-dir', default='.')
    parser.add_argument('config')

    return parser.parse_args()


if __name__ == '__main__':
    # setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


    args = parse_arguments()
    # load configuration
    config = yaml.load(open(args.config))

    n_epochs = args.n_epochs
    batch_size = 100
    seq_length = 20
    skip = 10
    nt = 640

    # set up meters
    meter_loss = tnt.meter.AverageValueMeter()
    meter_avg_loss = tnt.meter.AverageValueMeter()

    # open training data
    paths = config['paths']

    # get training loader
    def post(x):
        return x.isel(y=slice(24, 40))

    logger.info("Opening Training Data")
    train_data = get_dataset(paths, post=post)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # switch to output directory
    logger.info(f"Saving outputs in {args.output_dir}")
    try:
        os.mkdir(args.output_dir)
    except OSError:
        pass

    os.chdir(args.output_dir)

    # compute standard deviation
    logger.info("Computing Standard Deviation")
    scale = train_data.scale
    # compute scaler
    logger.info("Computing Mean")
    mean = train_data.mean

    cls = model.MLP

    logger.info(f"Training with {cls}")

    # restart
    if args.restart:
        logger.info(f"Restarting from checkpoint at {args.restart}")
        d = torch.load(args.restart)
        lstm = cls.from_dict(d['dict'])
        i_start = d['epoch'] + 1
    else:

        # initialize model
        lstm = cls(mean, scale, inputs=config['inputs'],
                   outputs=config['outputs'])
        i_start = 0

    logger.info(f"Training with {lstm}")

    # initialize optimizer
    optimizer = torch.optim.Adam(lstm.parameters(), lr=args.lr)

    try:
        for i in range(i_start, n_epochs):
            logging.info(f"Epoch {i}")
            for k, batch in enumerate(train_loader):
                logging.info(f"Batch {k} of {len(train_loader)}")
                n = get_batch_size(batch)

                # need to remove these auxiliary variables
                batch.pop('p')
                dm = batch.pop('layer_mass').detach_()
                loss = 0.0

                # set up loss function
                criterion = MVLoss(dm[0, :], config['loss_scale'])

                for t in range(0, nt - seq_length, skip):

                    # select window
                    window = select_time(batch, slice(t, t + seq_length))
                    x = select_time(window, slice(0, -1))
                    y = select_time(window, slice(1, None))

                    # make prediction
                    pred = lstm(x, n=1)

                    # compute loss
                    loss = criterion(window, pred)

                    # Back propagate
                    optimizer.zero_grad()
                    loss.backward()

                    # take step
                    optimizer.step()

                    # Log the results
                    meter_avg_loss.add(
                        criterion(window, mean).item())
                    meter_loss.add(loss.item())


                logger.info(f"Batch {k},  Loss: {meter_loss.value()[0]};"
                            f" Avg {meter_avg_loss.value()[0]}")
                meter_loss.reset()
                meter_avg_loss.reset()

            logger.info(f"Saving checkpoint to {i}.pkl")
            torch.save({'epoch': i, 'dict': lstm.to_dict()}, f"{i}.pkl")

    except KeyboardInterrupt:
        torch.save({'epoch': i, 'dict': lstm.to_dict()}, "interrupt.pkl")