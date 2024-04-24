import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import torch.nn.functional as F
from abc import abstractmethod
from numpy import inf
import time
import copy
import numpy as np
def prepare_device(device="gpu"):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    if device == "gpu":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("Warning: MPS device not found." "Training will be performed on CPU.")
            device = torch.device("cpu")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        raise NotImplementedError

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, soi_input, analog_input, soi_output, analog_output):
        assert len(soi_input) == len(soi_output) and len(analog_input) == len(analog_output), "Lengths of arrays must be the same"
        
        self.soi_input = torch.tensor(soi_input, dtype=torch.float32)
        self.analog_input = torch.tensor(analog_input, dtype=torch.float32)
        self.soi_output = torch.tensor(soi_output, dtype=torch.float32)
        self.analog_output = torch.tensor(analog_output, dtype=torch.float32)
        #self.mse = torch.tensor(np.mean((y1 - y2)**2, axis=1), dtype=torch.float32)  # Calculate MSE
        
    def __len__(self):
        return min(len(self.soi_input),len(self.analog_input))
    
    def __getitem__(self, idx):
        return self.soi_input[idx], self.analog_input[idx], torch.mean((self.soi_output[idx] - self.analog_output[idx])**2)
    
class TorchModel_base(nn.Module):
    def __init__(self, settings, input_dim1):
        super(TorchModel_base, self).__init__()
        self.input_dim1 = input_dim1
        self.settings = settings
        self.conv1 = torch.nn.Conv2d(input_dim1[-1], 16, 3)
        self.conv2 = torch.nn.Conv2d(16,32, 3)
        self.conv3 = torch.nn.Conv2d(32, 64, 3)
        self.conv4 = torch.nn.Conv2d(64, 128, 3)
        self.dropout_cnn = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.2)

        height, width = self._calculate_conv_output_dim(input_dim1)
        self.lin_cnn = nn.Linear(height*width*128, 16)
        # Define layers for x1
        self.lin_1 = nn.Linear(16, 16)
        self.lin_2 = nn.Linear(16, 16)

        self.map = nn.Linear(16, np.product(input_dim1), bias=False)
        nn.init.ones_(self.map.weight)
        for param in self.map.parameters():
            param.requires_grad = False
        self.bias_only = nn.Parameter(torch.ones(np.product(input_dim1)))

        # Define layers for x2
        self.scaler_train = nn.Linear(1, 1)
    def _calculate_conv_output_dim(self, input_dim):
        x = torch.randn(1, *input_dim)
        x = x.permute(0, 3, 1, 2)  # Change from NHWC to NCHW
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        _, _, height, width = x.shape
        return height, width
    def forward(self, x1, x2):
        if self.settings["cnn"] == 1:
            x1_cnn = x1.permute(0, 3, 1, 2)
            x = F.max_pool2d(F.relu(self.conv1(x1_cnn)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
            x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
            x = x.reshape(x.size(0), -1)
            x = F.relu(self.lin_cnn(x))
            x = F.relu(self.lin_1(self.dropout_fc(x)))
            x = F.relu(self.lin_2(self.dropout_fc(x)))
            x = self.map(x) + self.bias_only
        else:
            x = x1
            x = self.map(x) + self.bias_only
        new_size = (x.size()[0],) + self.input_dim1
        map = x.reshape(new_size)
        map = map / map.mean(dim=(1, 2, 3), keepdim=True)
        soi_weighted = map * x1
        analog_weighted = map * x2
        diff = ((torch.square((soi_weighted - analog_weighted)).sum(dim=(1,2,3)))) / np.product(self.input_dim1)
        diff = diff.unsqueeze(1)
        output = self.scaler_train(diff)
        return output, map

class TorchModel_gate(nn.Module):
    def __init__(self, settings, input_dim1, map_array = []):
        super(TorchModel_gate, self).__init__()
        self.input_dim1 = input_dim1
        self.map_array = map_array
        self.settings = settings
        self.conv1 = torch.nn.Conv2d(input_dim1[-1], 16, 3)
        self.conv2 = torch.nn.Conv2d(16,32, 3)
        self.conv3 = torch.nn.Conv2d(32, 64, 3)
        self.conv4 = torch.nn.Conv2d(64, 128, 3)
        self.dropout_cnn = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.2)

        height, width = self._calculate_conv_output_dim(input_dim1)
        self.lin_cnn = nn.Linear(height*width*128, 16)
        # Define layers for x1
        self.lin_1 = nn.Linear(16, 16)
        self.lin_2 = nn.Linear(16, 16)
        self.g = nn.Linear(16, len(self.map_array))

        self.branches = torch.Tensor(self.map_array.reshape((len(self.map_array),np.product(input_dim1))))

        # Define layers for x2
        self.scaler_train = nn.Linear(1, 1)
    def _calculate_conv_output_dim(self, input_dim):
        x = torch.randn(1, *input_dim)
        x = x.permute(0, 3, 1, 2)  # Change from NHWC to NCHW
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        _, _, height, width = x.shape
        return height, width
    def forward(self, x1, x2):
        x1_cnn = x1.permute(0, 3, 1, 2)
        x = F.max_pool2d(F.relu(self.conv1(x1_cnn)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.lin_cnn(x))
        x = F.relu(self.lin_1(self.dropout_fc(x)))
        x = F.relu(self.lin_2(self.dropout_fc(x)))
        gate = self.g(self.dropout_fc(x))
        T = self.settings["temperature"] #this is actually inverse temperature as T increases, get harder max
        gate = gate*T
        gate = F.softmax(gate)
        x = torch.mul(gate,self.branches)
        new_size = (x.size()[0],) + self.input_dim1
        map = x.reshape(new_size)
        map = map / map.mean(dim=(1, 2, 3), keepdim=True)
        soi_weighted = map * x1
        analog_weighted = map * x2
        diff = ((torch.square((soi_weighted - analog_weighted)).sum(dim=(1,2,3)))) / np.product(self.input_dim1)
        diff = diff.unsqueeze(1)
        output = self.scaler_train(diff)
        return output, map, gate

class MetricTracker:
    def __init__(self, *keys):

        self.history = dict()
        for k in keys:
            self.history[k] = []
        self.reset()

    def reset(self):
        for key in self.history:
            self.history[key] = []

    def update(self, key, value):
        if key in self.history:
            self.history[key].append(value)

    def result(self):
        for key in self.history:
            self.history[key] = np.nanmean(self.history[key])

    def print(self, idx=None):
        for key in self.history.keys():
            if idx is None:
                print(f"  {key} = {self.history[key]:.5f}")
            else:
                print(f"  {key} = {self.history[key][idx]:.5f}")

class BaseTrainer:
    """
    Base class for all trainers.
    """

    def __init__(
        self,
        model,
        criterion,
        metric_funcs,
        optimizer,
        max_epochs,
        settings
    ):

        self.settings = settings

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.max_epochs = max_epochs
        self.early_stopper = EarlyStopping(settings["patience"], settings["min_delta"])

        self.metric_funcs = metric_funcs
        self.batch_log = MetricTracker(
            "batch",
            "loss",
            "val_loss",
            *[m.__name__ for m in self.metric_funcs],
            *["val_" + m.__name__ for m in self.metric_funcs],
        )
        self.log = MetricTracker(
            "epoch",
            "loss",
            "val_loss",
            *[m.__name__ for m in self.metric_funcs],
            *["val_" + m.__name__ for m in self.metric_funcs],
        )

    def fit(self):
        """
        Full training logic
        """

        for epoch in range(self.max_epochs + 1):

            start_time = time.time()

            self._train_epoch(epoch)

            # log the results of the epoch
            self.batch_log.result()
            self.log.update("epoch", epoch)
            for key in self.batch_log.history:
                self.log.update(key, self.batch_log.history[key])

            # early stopping
            if self.early_stopper.check_early_stop(epoch, self.log.history["val_loss"][epoch], self.model):
                print(
                    f"Restoring model weights from the end of the best epoch {self.early_stopper.best_epoch}: "
                    f"val_loss = {self.early_stopper.min_validation_loss:.5f}"
                )
                self.log.print(idx=self.early_stopper.best_epoch)

                self.model.load_state_dict(self.early_stopper.best_model_state)
                self.model.eval()

                break

            # Print out progress during training
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"Epoch {epoch:3d}/{self.max_epochs:2d}\n"
                f"  {elapsed_time:.1f}s"
                f" - train_loss: {self.log.history['loss'][epoch]:.5f}"
                f" - val_loss: {self.log.history['val_loss'][epoch]:.5f}"
            )

        # reset the batch_log
        self.batch_log.reset()

    @abstractmethod
    def _train_epoch(self):
        """
        Train an epoch

        """
        raise NotImplementedError

    @abstractmethod
    def _validation_epoch(self):
        """
        Validate after training an epoch

        """
        raise NotImplementedError


class EarlyStopping:
    """
    Base class for early stopping.
    """

    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")
        self.best_model_state = None
        self.best_epoch = None

    def check_early_stop(self, epoch, validation_loss, model):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0

            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
    
class MaskTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_funcs,
        optimizer,
        max_epochs,
        data_loader,
        validation_data_loader,
        device,
        settings,
    ):
        super().__init__(
            model,
            criterion,
            metric_funcs,
            optimizer,
            max_epochs,
            settings,
        )
        self.settings = settings
        self.device = device

        self.data_loader = data_loader
        self.validation_data_loader = validation_data_loader

        self.do_validation = True

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.batch_log.reset()

        for batch_idx, (soi, analog, target) in enumerate(self.data_loader):
            soi_input, analog_input, target = (
                soi.to(self.device),
                analog.to(self.device),
                target.to(self.device),
            )

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            output, map = self.model(soi_input, analog_input)
            # output = self.model(input)

            # Compute the loss and its gradients
            loss = self.criterion(output, target)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Log the results
            self.batch_log.update("batch", batch_idx)
            self.batch_log.update("loss", loss.item())
            for met in self.metric_funcs:
                self.batch_log.update(met.__name__, met(output, target))

        # Run validation
        if self.do_validation:
            self._validation_epoch(epoch)

    def _validation_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():

            for batch_idx, (soi, analog, target) in enumerate(self.validation_data_loader):
                soi_input, analog_input, target = (
                soi.to(self.device),
                analog.to(self.device),
                target.to(self.device),
            )

                output, map = self.model(soi_input, analog_input)
                loss = self.criterion(output, target)

                # Log the results
                self.batch_log.update("val_loss", loss.item())
                for met in self.metric_funcs:
                    self.batch_log.update("val_" + met.__name__, met(output, target))

class GateTrainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metric_funcs,
        optimizer,
        max_epochs,
        data_loader,
        validation_data_loader,
        device,
        settings,
    ):
        super().__init__(
            model,
            criterion,
            metric_funcs,
            optimizer,
            max_epochs,
            settings,
        )
        self.settings = settings
        self.device = device

        self.data_loader = data_loader
        self.validation_data_loader = validation_data_loader

        self.do_validation = True

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        self.model.train()
        self.batch_log.reset()

        for batch_idx, (soi, analog, target) in enumerate(self.data_loader):
            soi_input, analog_input, target = (
                soi.to(self.device),
                analog.to(self.device),
                target.to(self.device),
            )

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            output, map, gate = self.model(soi_input, analog_input)
            # output = self.model(input)

            # Compute the loss and its gradients
            loss = self.criterion(output, target)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Log the results
            self.batch_log.update("batch", batch_idx)
            self.batch_log.update("loss", loss.item())
            for met in self.metric_funcs:
                self.batch_log.update(met.__name__, met(output, target))

        # Run validation
        if self.do_validation:
            self._validation_epoch(epoch)

    def _validation_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():

            for batch_idx, (soi, analog, target) in enumerate(self.validation_data_loader):
                soi_input, analog_input, target = (
                soi.to(self.device),
                analog.to(self.device),
                target.to(self.device),
            )

                output, map, gate = self.model(soi_input, analog_input)
                loss = self.criterion(output, target)

                # Log the results
                self.batch_log.update("val_loss", loss.item())
                for met in self.metric_funcs:
                    self.batch_log.update("val_" + met.__name__, met(output, target))

