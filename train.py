from torch.utils.data import DataLoader
import torch
from utils import metrics_and_vizulatisation


def train(model, dataset, loss, optimizer, device, scheduler, args):
    """
    Train the model and store the results
    """
    
    # Create dataloader
    data_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=True, 
                      drop_last=True, num_workers=4)
    
    # Indicate that we are in training mode
    model.train()
    mean_losses = []
    for epoch in range(args['epochs']):
        losses = []
        for i, batch in enumerate(data_loader):
            # Get the batches.
            audio_features, label = batch[0], batch[1]

            #metrics_and_vizulatisation.plot_histograms(audio_features[1,:,:].squeeze(0).flatten())

            # Give them to the appropriate device.
            audio_features = audio_features.to(device)
            label = label.to(device)

            # Unsqueeze as the network requires it to be certain dimension in order to batch process
            audio_features = audio_features.unsqueeze(1)

            # Reconstructed audio
            reconstructed_audio = model(audio_features)


            train_loss = loss(reconstructed_audio, audio_features)
            train_loss.backward()

            optimizer.step()

            losses.append(train_loss.detach().cpu().numpy())
            print(f"{i}/{len(data_loader)}")

        mean_losses.append(sum(losses)/len(losses))
        scheduler.step(sum(losses)/len(losses))
        print(f"Epoch {epoch} average loss: {sum(losses)/len(losses)}")
        print(optimizer.param_groups[0]['lr'])
        # Store weights
        torch.save(model.state_dict(), f'weights/latest{args["audio_features_type"]}{args["object_id"]}Audio.pth')
    metrics_and_vizulatisation.plot_loss(mean_losses)