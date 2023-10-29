import tqdm
import torch
import torch.nn.functional as F

class Trainer():
    def __init__(self, model, optimizer, criterion, lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

    def train(self, train_data_loader, test_data_loader=None, eval=False, device='cpu', n_epoches=10):
        if (eval == True and test_data_loader is None):
            raise ValueError("<test_data_loader> must be provided for evaluation")

        for epoch in range(n_epoches):
            running_loss = 0.0
            self.model.train()
            for batch in tqdm.tqdm(train_data_loader, desc="Train"):
                self.optimizer.zero_grad()

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
            print(f'[Epoch {epoch + 1}] Loss: {running_loss}')

            if (eval):
                self.test(test_data_loader, device)

    def test(self, test_data_loader, device='cpu'):
        accuracy = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm.tqdm(test_data_loader, desc="Test"):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.model(inputs)
                outputs = F.softmax(outputs, dim=1).argmax(dim=1)
                accuracy += torch.sum(outputs == labels)
        accuracy /= len(test_data_loader.dataset)
        print(f'Accuracy: {accuracy}')