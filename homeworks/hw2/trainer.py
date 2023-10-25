import tqdm
import torch
import torch.nn.functional as F

class Trainer():
    def __init__(self, model, optimizer, criterion, lr_scheduler=None, n_epoches=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.n_epoches = n_epoches

    def train(self, train_data_loader, test_data_loader=None, eval=False, device='cpu'):
        if (eval == True and test_data_loader is None):
            raise ValueError("<test_data_loader> must be provided for evaluation")

        for epoch in range(self.n_epoches):
            running_loss = 0.0
            self.model.train()
            for batch in tqdm.tqdm(train_data_loader, desc="Train"):
                self.optimizer.zero_grad()
                print(batch.shape)
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()
                if (self.lr_scheduler):
                    self.lr_scheduler.step()

                running_loss += loss.item()
            running_loss /= len(train_data_loader)
            print(f'[Epoch {epoch + 1}] Loss: {running_loss}', end='')

            if (eval):
                accuracy = 0.0
                self.model.eval()
                with torch.no_grad():
                    for batch in tqdm.tqdm(test_data_loader, desc="Test"):
                        print(batch.shape)
                        inputs, labels = batch
                        inputs, labels = inputs.to(device), labels.to(device)

                        outputs = self.model(inputs)
                        outputs = F.softmax(outputs, dim=1).argmax(dim=1)
                        accuracy += torch.sum(outputs == labels)
                accuracy /= len(test_data_loader.dataset)
                print(f', Accuracy: {accuracy}', end='')

            print('.')