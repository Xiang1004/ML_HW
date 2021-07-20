import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import matplotlib.pyplot as plt
from model import ConvNet, MyNet
from data import get_dataloader
import numpy as np

if __name__ == "__main__":
    # Specifiy data folder path and model type(fully/conv)
    # folder, model_type = sys.argv[1], sys.argv[2]

    folder = r"C:\Users\Xiang\Desktop\ML_HW\hw2\hw2_data\p2"
    model_type = "conv"

    # Get data loaders of training set and validation set
    train_loader, val_loader = get_dataloader(folder, batch_size=32)

    # Specify the type of model
    if model_type == 'conv':
        model = ConvNet()
    elif model_type == 'mynet':
        model = MyNet()

    # Set the type of gradient optimizer and the model it update
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Choose loss function
    criterion = nn.CrossEntropyLoss()

    # Check if GPU is available, otherwise CPU is used
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    # Run any number of epochs you want
    ep = 10
    eval_acc = []
    for epoch in range(ep):
        print('Epoch:', epoch)
        ##############
        ## Training ##
        ##############

        # Record the information of correct prediction and loss
        correct_cnt, total_loss, total_cnt = 0, 0, 0

        # Load batch data from dataloader
        for batch, (x, label) in enumerate(train_loader, 1):
            # Set the gradients to zero (left by previous iteration)
            optimizer.zero_grad()
            # Put input tensor to GPU if it's available
            if use_cuda:
                x, label = x.cuda(), label.cuda()
            # Forward input tensor through your model
            out = model(x)
            # Calculate loss
            loss = criterion(out, label)

            # Compute gradient of each model parameters base on calculated loss
            loss.backward()
            # Update model parameters using optimizer and gradients
            optimizer.step()

            # Calculate the training loss and accuracy of each iteration
            total_loss += loss.item()
            _, pred_label = torch.max(out, 1)
            total_cnt += x.size(0)
            correct_cnt += (pred_label == label).sum().item()

            # Show the training information
            if batch % 500 == 0 or batch == len(train_loader):
                acc = correct_cnt / total_cnt
                ave_loss = total_loss / batch
                print('Training batch index: {}, train loss: {:.6f}, acc: {:.3f}'.format(
                    batch, ave_loss, acc))
        ################
        ## Validation ##
        ################

        model.eval()
        # TODO
        v_correct_cnt, v_total_loss, v_total_cnt = 0, 0, 0
        with torch.no_grad():

            for v_batch, (x, label) in enumerate(val_loader, 1):
                val_out = model(x)
                val_loss = criterion(val_out, label)

                v_total_loss += val_loss.item()
                _, v_pred_label = torch.max(val_out, 1)
                v_total_cnt += x.size(0)
                v_correct_cnt += (v_pred_label == label).sum().item()

                if v_batch % 500 == 0 or v_batch == len(val_loader):
                    v_acc = v_correct_cnt / v_total_cnt
                    v_ave_loss = v_total_loss / v_batch
                    print('Eval batch index: {}, eval loss: {:.6f}, eval_acc: {:.3f}'.format(v_batch, v_ave_loss, v_acc))
        eval_acc.append(v_acc)

        model.train()
    print(eval_acc)
    # Save trained model
    torch.save(model.state_dict(), './checkpoint/%s.pth' % model.name())

    # Plot Learning Curve
    # TODO
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    plt.plot(list(range(ep)), eval_acc, 'b-')
    plt.title('Learning Curve')
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()