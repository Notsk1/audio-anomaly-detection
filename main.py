
import arguments
import train
import test
import pathlib
from torch.utils.data import DataLoader
import torch
from datasets.dataset import AnomalyDataset
from models.mel_model import AnomalyMelCNN
from torchinfo import summary


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    data_folder = pathlib.Path(args['data_folder'])

    dataset = AnomalyDataset(args['split_type'], data_folder, args['object_id'], args['load_into_memory'])

    model = AnomalyMelCNN(dense=args['dense'], skip=args['skip'])

    if args['split_type'] == 'training':
        reduction = 'mean'
    else:
        reduction = 'none'

    loss = torch.nn.MSELoss(reduction = reduction)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3)

    model.to(device)
    summary(model, input_size=(args['batch_size'], 1, 128, 312))
    if args['split_type'] == 'training':
        if args['load_training']:
            model.load_state_dict(torch.load(f'weights/latest{args["audio_features_type"]}Audio.pth'))
        train.train(model, dataset, loss, optimizer, device, scheduler, args)

    elif args['split_type'] == 'validation':
        model.load_state_dict(torch.load(f'weights/latest{args["audio_features_type"]}{args["object_id"]}Audio.pth'))
        test.test(model, dataset, loss, device, args)

    elif args['split_type'] == 'testing':
        model.load_state_dict(torch.load(f'weights/latest{args["audio_features_type"]}{args["object_id"]}Audio.pth'))
        test.test(model, dataset, loss, device, args)

    else:
        raise ValueError("Invalid split type")
        

if __name__ == '__main__':
    args = arguments.getArgs()
    main(args)