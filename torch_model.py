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
import matplotlib.pyplot as plt
import matplotlib as mpl
import base_directories
import cartopy as ct


dir_settings = base_directories.get_directories()
if dir_settings["data_directory"].split("/")[1] != "Users":
    ct.config["data_dir"] = "/scratch/jlandsbe/cartopy_maps"
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["figure.dpi"] = 150
#plt.style.use("seaborn-v0_8")
dpiFig = 300

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
    def __init__(self, analog_input, soi_input, analog_output, soi_output, tether_analogs=[], tether_soi=[], validation_soi_indices = None, validation_analog_indices = None, rng_seed=33):
        #assert len(soi_input) == len(soi_output), "Lengths of SOI arrays must be the same"
        #assert len(analog_input) == len(analog_output), "Lengths of analog arrays must be the same"
        self.validation_soi_indices = validation_soi_indices
        self.validation_analog_indices = validation_analog_indices
        self.soi_input = torch.tensor(soi_input, dtype=torch.float32)
        self.analog_input = torch.tensor(analog_input, dtype=torch.float32)
        self.soi_output = torch.tensor(soi_output, dtype=torch.float32)
        self.analog_output = torch.tensor(analog_output, dtype=torch.float32)
        self.analog_tethers = torch.tensor(tether_analogs, dtype=torch.float32)
        self.soi_tethers = torch.tensor(tether_soi, dtype=torch.float32)
        if tether_analogs != [] and tether_soi!=[]:
            self.analog_all_tethers = torch.cat((self.analog_output.unsqueeze(0), self.analog_tethers), dim=0)
            self.soi_all_tethers = torch.cat((self.soi_output.unsqueeze(0), self.soi_tethers), dim=0)
        torch.manual_seed(rng_seed)
                # Calculate min and max of soi_output for normalization
        self.soi_output_min = self.soi_output.min()
        self.soi_output_max = self.soi_output.max()
        
    def __len__(self):
        return max(len(self.soi_input),len(self.analog_input))
        #return 10000
    
    def __getitem__(self, idx):
        if self.validation_analog_indices is not None and self.validation_soi_indices is not None:
            analog_idx = self.validation_analog_indices[idx]
            soi_idx = self.validation_soi_indices[idx]
        else:
            analog_idx = int(torch.randint(0, len(self.analog_input), (1,)).item())
            soi_idx = int(torch.randint(0, len(self.soi_input), (1,)).item())
        if self.analog_tethers.shape[0] > 0:
            err = torch.mean((self.soi_all_tethers[:,soi_idx] - self.analog_all_tethers[:,analog_idx]**2))
        else:
            soi_out = self.soi_output[soi_idx]
            soi_out_normalized = 2 * (soi_out - self.soi_output_min) / (self.soi_output_max - self.soi_output_min)
            err = torch.mean((self.soi_output[soi_idx] - self.analog_output[analog_idx])**2)
        return self.soi_input[soi_idx], self.analog_input[analog_idx], err, soi_out
    
class TorchModel_base(nn.Module):
    def __init__(self, settings, input_dim1):
        super(TorchModel_base, self).__init__()
        self.input_dim1 = input_dim1
        self.settings = settings
        self.conv1 = torch.nn.Conv2d(input_dim1[-1], 16, 3)
        self.conv2 = torch.nn.Conv2d(16,32, 3)
        self.conv3 = torch.nn.Conv2d(32, 64, 3)
        self.conv4 = torch.nn.Conv2d(64, 128, 3)
        self.dropout_cnn = nn.Dropout(0.1)
        self.dropout_fc = nn.Dropout(0.1)

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
                # Define parameters for positive weight and bias
        self.scaler_train_weight_param = nn.Parameter(torch.ones(1))  # Parameter to be transformed to positive weight
        self.scaler_train_bias_param = nn.Parameter(torch.ones(1))    
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
            new_size = (x.size()[0],) + self.input_dim1
            map = x.reshape(new_size)
            map = F.relu(map)
            map = map / map.mean(dim=(1, 2, 3), keepdim=True) #comment out this line
        else:
            map = self.bias_only.reshape(self.input_dim1) 
            map = F.relu(map)
            map = map / map.mean()
        soi_weighted = map * x1
        analog_weighted = map * x2
        diff = ((torch.square((soi_weighted - analog_weighted)).sum(dim=(1,2,3)))) / np.product(self.input_dim1)
        #return diff, map
        diff = diff.unsqueeze(1)
        positive_scaling = torch.square(self.scaler_train_weight_param)
        output = positive_scaling * diff + self.scaler_train_bias_param
        output = self.scaler_train(diff)
        return output, map #comment out this and instead return diff, map



#come up with gate
class TorchModel_gate(nn.Module):
    def __init__(self, settings, input_dim1, map_array = []):
        super(TorchModel_gate, self).__init__()
        self.input_dim1 = input_dim1
        self.map_array = map_array
        self.settings = settings
        self.conv1 = torch.nn.Conv2d(input_dim1[-1], 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(16,32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = torch.nn.Conv2d(64, 128, 3)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout_cnn = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.2)

        height, width = self._calculate_conv_output_dim(input_dim1)
        self.lin_cnn = nn.Linear(height*width*128, 16)
        # Define layers for x1
        self.lin_1 = nn.Linear(16, 16)
        self.lin_2 = nn.Linear(16, 16)
        self.g = nn.Linear(16, len(self.map_array))
        self.gates = self.gates = 1/len(self.map_array)* torch.ones(len(self.map_array))

    def _calculate_conv_output_dim(self, input_dim):
        x = torch.randn(1, *input_dim)
        x = x.permute(0, 3, 1, 2)  # Change from NHWC to NCHW
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        _, _, height, width = x.shape
        return height, width
    def forward(self, x1):
        x1_cnn = x1.permute(0, 3, 1, 2)
        x1_cnn = x1.permute(0, 3, 1, 2)
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x1_cnn))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), (2, 2))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), (2, 2))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.lin_cnn(x))
        x = F.relu(self.lin_1(self.dropout_fc(x)))
        x = F.relu(self.lin_2(self.dropout_fc(x)))
        gate = self.g(self.dropout_fc(x))
        T = self.settings["temperature"] #this is actually inverse temperature as T increases, get harder max
        if self.training:
            noise_factor = .0*gate.mean().item()  # Adjust this to add more or less noise
            noise = torch.randn_like(gate) * noise_factor
            gate = gate + noise
        gate = gate*T
        gate = F.softmax(gate)
        self.gates = gate
        return self.gates

#given a gate, pick the map an evaluate loss
class TorchModel_gatedMap(nn.Module):
    def __init__(self, settings, input_dim1, gates, map_array = []):
        super(TorchModel_gatedMap, self).__init__()
        self.input_dim1 = input_dim1
        self.map_array = map_array
        self.settings = settings
        self.gate =  gates
        self.branches = torch.Tensor(self.map_array.reshape((len(self.map_array),np.product(input_dim1))))

        # Define layers for x2
        self.scaler_train = nn.Linear(1, 1)
    def forward(self, x1, x2):
        x = torch.matmul(self.gate.view(1, -1),self.branches)
        new_size = (x.size()[0],) + self.input_dim1
        map = x.reshape(new_size)
        map = map / map.mean(dim=(1, 2, 3), keepdim=True)
        soi_weighted = map * x1
        analog_weighted = map * x2
        diff = ((torch.square((soi_weighted - analog_weighted)).sum(dim=(1,2,3)))) / np.product(self.input_dim1)
        diff = diff.unsqueeze(1)
        output = self.scaler_train(diff)
        #output = torch.matmul(output, self.gate.view(-1, 1))
        return output, map
    
class CombinedGateModel(nn.Module):
    def __init__(self, settings, input_dim1, map_array=[]):
        super(CombinedGateModel, self).__init__()
        self.gate_model = TorchModel_gate(settings, input_dim1, map_array)
        self.map_model = TorchModel_gatedMap(settings, input_dim1, self.gate_model.gates, map_array)

    def forward(self, x1, x2):
        gate = self.gate_model(x1)
        output, map = self.map_model(x1, x2)
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
                    f"val_loss = {self.early_stopper.min_validation_loss:.5f}", flush=True
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
                f" - val_loss: {self.log.history['val_loss'][epoch]:.5f}", flush=True
            )

        # reset the batch_log
        self.batch_log.reset()

    def normalize(self,values):
        min_val = min(values)
        max_val = max(values)
        return [(x - min_val) / (max_val - min_val) for x in values]
    
    def plot_loss(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.log.history['loss'], label='Train Loss')
        plt.plot(self.log.history['val_loss'], label='Validation Loss')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.tight_layout()
        print(dir_settings["figure_diag_directory"] + self.settings["savename_prefix"] +
                '_training_history.png')
        plt.savefig(dir_settings["figure_diag_directory"] + self.settings["savename_prefix"] +
                '_training_history.png', dpi=dpiFig, bbox_inches='tight')
        plt.close()

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

        for batch_idx, (soi, analog, target, soi_out) in enumerate(self.data_loader):
            if batch_idx >= int(self.settings["max_iterations"]/self.settings["batch_size"])-1:
                break
            soi_input, analog_input, target = (
                soi.to(self.device),
                analog.to(self.device),
                target.to(self.device),
            )

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            output, map = self.model(soi_input, analog_input)


            # Compute the loss and its gradients
            target = target.view(-1, 1)
            soi_out = soi_out.view(-1, 1)
            losses = self.criterion(output, target)
            if self.settings["weighted_train"] != 0:
                weights =  ((1/(target + 1e-8))**self.settings["weighted_train"])
            else:
                weights = torch.ones_like(target)
            if self.settings["extremes_weight"]>0:
                if self.settings["extremes_percentile"] > 0:
                    weights = weights**self.settings["extremes_weight"] * soi_out
                elif self.settings["extremes_percentile"] < 0:
                    weights = weights**self.settings["extremes_weight"] * (1/soi_out)
            weights = weights/weights.mean()
            weighted_losses = losses * weights
            loss = weighted_losses.mean()
                # Add L1 regularization
            l1_lambda = self.settings["mask_l1"]  # You can adjust this value
            l2_inverse_lambda = self.settings["mask_l2_inverse"]
            l1_norm = torch.norm(map, p=1)
            l2_norm = torch.norm(map, p=2)
            loss += l1_lambda * l1_norm
            loss += l2_inverse_lambda / l2_norm
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
            for batch_idx, (soi, analog, target, soi_out) in enumerate(self.validation_data_loader):
                if batch_idx >= int(self.settings["max_iterations"]/self.settings["val_batch_size"])-1:
                    break
                soi_input, analog_input, target = (
                soi.to(self.device),
                analog.to(self.device),
                target.to(self.device),
            )

                # Make predictions for this batch
                output, map = self.model(soi_input, analog_input)

               

                # Compute the loss and its gradients
                target = target.view(-1, 1)
                soi_out = soi_out.view(-1, 1)
                losses = self.criterion(output, target)
                if self.settings["weighted_train"] != 0:
                    weights =  ((1/(target + 1e-8))**self.settings["weighted_train"])
                else:
                    weights = torch.ones_like(target)
                if self.settings["extremes_weight"]>0:
                    if self.settings["extremes_percentile"] > 0:
                        weights = weights**self.settings["extremes_weight"] * (soi_out > 1.0) *1.0
                    elif self.settings["extremes_percentile"] < 0:
                        weights = weights**self.settings["extremes_weight"] * (1/soi_out)
                weights = weights/weights.mean()
                weighted_losses = losses * weights
                loss = weighted_losses.mean()
                    # Add L1 regularization
                l1_lambda = self.settings["mask_l1"]  # You can adjust this value
                l2_inverse_lambda = self.settings["mask_l2_inverse"]
                l1_norm = torch.norm(map, p=1)
                l2_norm = torch.norm(map, p=2)
                loss += l1_lambda * l1_norm
                loss += l2_inverse_lambda / l2_norm

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

