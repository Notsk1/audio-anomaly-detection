from torch.utils.data import DataLoader
import torch
from utils import metrics_and_vizulatisation
import statistics

def test(model, dataset, loss, device, args):
    # Create dataloader
    data_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=False, 
                      drop_last=True, num_workers=4)
    
    # Indicate that we are in training mode
    model.eval()
    losses = []
    normal_losses = []
    anomaly_losses = []
    rightClasses = 0
    number_of_anomalies = 0
    for batch in data_loader:
        # Get the batches.
        audio_features, labels = batch[0], batch[1]

        # Give them to the appropriate device.
        audio_features = audio_features.to(device)
        labels = labels.to(device)

        # Unsqueeze as the network requires it to be certain dimension in order to batch process
        audio_features = audio_features.unsqueeze(1)

        print("Audio features: ", audio_features[0,:,:,:])
        # Reconstructed audio
        reconstructed_audio = model(audio_features)

        print("Reconstructed audio: ", reconstructed_audio[0,:,:,:])
        test_loss = loss(reconstructed_audio, audio_features)
        test_loss =  torch.mean(test_loss, dim=(2,3))
        test_loss.to(device)

        # Threshold the output:
        y_hat = torch.ones((args['batch_size'], 1))
        y_hat = y_hat.to(device)
        y_hat = torch.mul((test_loss <= 0.0001035), y_hat)

        # Accuracy:
        y_hat = torch.reshape(y_hat, (1, args['batch_size']))

        rightClasses += (y_hat == labels.type_as(y_hat)).sum().detach()

        # Loss for each class:
        bool_labels = labels.clone().type(torch.bool)

        normal_loss = test_loss[~bool_labels.unsqueeze(1)].tolist()
        anomaly_loss = test_loss[bool_labels.unsqueeze(1)].tolist()
        number_of_anomalies += labels.sum().detach()

        # Plot normalized Mel spectograms:
        #metrics_and_vizulatisation.plot_mel(audio_features, reconstructed_audio)

        # Add all losses to a list
        if normal_loss:
            normal_losses.extend(normal_loss)
        if anomaly_loss:
            anomaly_losses.extend(anomaly_loss)

        losses.append(test_loss.detach())

    auc = metrics_and_vizulatisation.calculate_AUC(anomaly_losses, normal_losses)
    metrics_and_vizulatisation.plot_histograms(anomaly_losses, normal_losses)


    print(f"AUC score: {auc}")
    print(f"Average normal loss: {sum(normal_losses)/len(normal_losses)}")
    print(f"Average anomaly loss: {sum(anomaly_losses)/len(anomaly_losses)}")
    print(f"Median normal loss: {statistics.median(normal_losses)}")
    print(f"Median anomaly loss: {statistics.median(anomaly_losses)}")
    print(f"Average combined loss: {sum(losses)/len(losses)}")
    print(f"Random choice accuracy: {number_of_anomalies/(len(data_loader)*args['batch_size'])}")
    print(f"Testing accuracy: {rightClasses/(len(data_loader)*args['batch_size'])}")

