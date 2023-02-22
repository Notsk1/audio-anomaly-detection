
import arguments
import train
import test
import pathlib
from torch.utils.data import DataLoader
from datasets.dataset import AnomalyDataset

def main(args):
    data_folder = pathlib.Path(args['data_folder'])

    dataset = AnomalyDataset(args['split_type'], data_folder, args['object_id'], args['load_into_memory'])

    model = 0
    loss = 0
    optimizer = 0
    if args['split_type'] == 'training':
        train.train(model, dataset, loss, optimizer, args)
    elif args['split_type'] == 'validation':
        test.validate()
    elif args['split_type'] == 'testing':
        test.test()
    else:
        raise ValueError("Invalid split type")
        

if __name__ == '__main__':
    args = arguments.getArgs()
    main(args)