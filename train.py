import test
from torch.utils.data import DataLoader


def train(model, dataset, loss, optimizer, args):
    """
    Train the model and store the results
    """
    # Create dataloader
    data_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=True, 
                      drop_last=True, num_workers=4)
    
    for batch in data_loader:
        # Get the batches.
        x, y = batch[0], batch[1]
        print(x,y)

