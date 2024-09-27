from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import os
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score
from Pyfhel import Pyfhel, PyCtxt
from typing import Callable, Dict, List, Optional, Tuple, Union
import tempfile
from collections import OrderedDict
from flwr.common import Parameters, Metrics
import copy
import ast
import re
import numpy as np
import torch
import math


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]
        
        if self.transform:
            img = self.transform(img)
        return img, label


def read_data(path):
    labels_path = path + '\\labels'
    images_path = path + '\\images'
    labels = []
    images = []
    mapping = {0: 2, 127: 0, 255: 1}
    hdf_files = [filename for filename in os.listdir(labels_path) if filename.endswith('.h5')]
    hdf_files.sort()

    png_filenames = [filename for filename in os.listdir(images_path) if filename.endswith('.png')]
    png_filenames.sort()

    for file in hdf_files:
        file_path = os.path.join(labels_path, file)
        with h5py.File(file_path, 'r') as f:
            matrix = np.array(f['Raster Image #0'])
            to_float = np.vectorize(lambda x: np.float32(mapping.get(x)))
            matrix = to_float(matrix)
            labels.append(matrix)

    for filename in png_filenames:
        image_path = os.path.join(images_path, filename)
        image = Image.open(image_path)
        image_array = np.array(image, np.float32)
        images.append(image_array)
    return labels, images


def load_datasets_centralized(test_size, val_size, batch_size, path):
    labels, images = read_data(path)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    test_size = val_size / (1 - test_size)
    x_train, X_train, y_train, Y_train = train_test_split(x_train, y_train, test_size=test_size, random_state=42)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),
        ]
    )
    trainset = ImageDataset(x_train, y_train, transform=transform)
    testset = ImageDataset(x_test, y_test, transform=transform)
    valset = ImageDataset(X_train, Y_train, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader, testloader


def load_datasets_federated(batch_size, num_clients, path):
    labels, images = read_data(path)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),
        ]
    )
    trainset = ImageDataset(images, labels, transform=transform)
    chunk_size = len(trainset) // num_clients
    lengths = [chunk_size] * (num_clients - 1) + [len(trainset) - chunk_size * (num_clients - 1)]

    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))
    trainloaders = []
    valloaders = []
    for ds in datasets:
        max_len = max(len(ds) // 10, 1) 
        len_train = len(ds) - max_len
        lengths = [len_train, max_len]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size, shuffle=True))
    print(f"Samples in train loaders:{sum([len(ds.dataset) for ds in trainloaders])}")
    print(f"Samples in val loaders:{sum([len(ds.dataset) for ds in valloaders])}")

    return trainloaders, valloaders


def train(net, trainloader, epochs, device, verbose=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)["out"]
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(trainloader)
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}")



def test(net, testloader, device):
    net.eval() 
    test_loss, correct, total, iou, f_score, accuracy = 0,0,0,0,0,0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)["out"]
            loss = criterion(outputs, labels.long())
            _, predicted = torch.max(outputs.data, 1)
            test_loss += loss.item()
            total += labels.numel()  
            labels = labels.cpu().numpy()
            predicted = predicted.cpu().numpy()
            correct += np.sum(predicted == labels)  
            iou += jaccard_score(labels.flatten(), predicted.flatten(), average='weighted')
            f_score += f1_score(labels.flatten(), predicted.flatten(), average='weighted')
    test_loss /= total
    accuracy = correct / total
    iou /= len(testloader)
    f_score /= len(testloader)
    return test_loss, iou, accuracy, f_score



def create_context(n = 2**15, scale = 2**30, qi_sizes=[60, 30, 60], sec=128):
    HE = Pyfhel()
    ckks_params = {'scheme': 'CKKS', 'n': n, 'scale': scale,'qi_sizes': [60, 30, 60], 'sec': sec}
    HE.contextGen(**ckks_params)  
    HE.keyGen()
    return HE


def flatten_parameters(ndarrays):
    shapes = [param.shape for param in ndarrays]
    flat_parameters = np.concatenate([param.flatten() for param in ndarrays])
    return flat_parameters, shapes


def encrypt(flat_params, HE):
    batch_size = HE.get_nSlots()
    num_batches = math.ceil(len(flat_params)/ batch_size)
    res = []

    for i in range(num_batches):
        a = i * batch_size
        b = (i + 1) * batch_size
        batch = flat_params[a:b]
        encrypted_batch = HE.encryptFrac(batch)
        res.append(encrypted_batch)

    return res

def decrypt(encrypted_params, HE, original_shapes):
    flat_params = []
    for i in range(len(encrypted_params)):
        decrypted_params = HE.decryptFrac(encrypted_params[i])
        flat_params.extend(decrypted_params)

    num_elements = int(sum(np.prod(shape) if shape else 1 for shape in original_shapes))
    params = flat_params[:num_elements]
    res = []
    i = 0
    for shape in original_shapes:
        if shape:  
            size = int(np.prod(shape))
            batch = np.array(params[i:i+size])
            reshaped = batch.reshape(shape)
            res.append(reshaped)
            i += size
        else:
            val = np.array([round(params[i])])
            res.append(val)
            i += 1
    return res

def aggregate_encrypted(params_lists, HE):
    num_clients = len(params_lists)
    num_batches = len(params_lists[0])

    inverse_num_clients = HE.encode(1.0 / num_clients)
    aggregated_batches = params_lists[0]

    for client_batches in params_lists[1:]:
        for i, batch in enumerate(client_batches):
            aggregated_batches[i] += batch

    for i in range(num_batches):
        aggregated_batches[i] *= inverse_num_clients

    return aggregated_batches


def serialize(encrypted_params):
    batches = []
    for batch in encrypted_params:
        batches.append(batch.to_bytes())
    return Parameters(tensors=batches, tensor_type="PYCtxt")

def deserialize(parameter_object, HE):
    batches = []
    for batch in parameter_object.tensors:
        batches.append(PyCtxt(bytestring=batch, pyfhel=HE))
    return batches

def train_early_stopping(model, trainloader, testloader, valloader, epochs, device, patience):
    best_loss = float('inf')
    no_improvement = 0
    results = {'Loss': [],
               'IoU': [],
               'Accuracy': [],
               'F-score': [],
               }

    for epoch in range(epochs):
        train(model, trainloader, 1, device=device)
        val_loss, val_iou, val_accuracy, val_f_score = test(model, valloader, device=device)
        results['Loss'].append(val_loss)
        results['IoU'].append(val_iou)
        results['Accuracy'].append(val_accuracy)
        results['F-score'].append(val_f_score)

        if val_loss < best_loss:
            best_loss = val_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement == patience:
                print("Stopping.")
                break
    
    test_loss, test_iou, test_accuracy, test_f_score = test(model, testloader, device=device)
    return results

def train_models(models, trainloader, valloader, epochs, device):
    results = {}
    for name, model in models.items():
        for epoch in range(epochs):
            train(model, trainloader, epochs=1, device=device)
            val_loss, val_iou, val_accuracy, val_f_score = test(model, valloader, device=device)

            print(f"Epoch {epoch+1}:validation loss {val_loss}\n\t val iou {val_iou}\n\t accuracy {val_accuracy}")

            if name not in results:
                results[name] = {'val_loss': [], 'val_iou': [], 'val_accuracy': []}

            results[name]['val_loss'].append(val_loss)
            results[name]['val_iou'].append(val_iou)
            results[name]['val_accuracy'].append(val_accuracy)
    return results


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    state_dict = {}
    params_dict = zip(net.state_dict().keys(), parameters)
    for k, v in params_dict:
        state_dict[k] = torch.from_numpy(np.copy(v)).view(net.state_dict()[k].shape)
    net.load_state_dict(state_dict, strict=True)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    iou_list = [num_examples * m["iou"] for num_examples, m in metrics]
    accuracy_list = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f_score_list = [num_examples * m["f_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"IoU": sum(iou_list) / sum(examples),
            "Accuracy": sum(accuracy_list) / sum(examples),
            "F-Score": sum(f_score_list) / sum(examples)}


def parse_results(results):
    all_metrics = []
    for i in range(len(results)):
        s = results[i].split('):\n')
        metrics_string = s[2]
        metrics_dict = ast.literal_eval(metrics_string)
        iou_values = []
        accuracy_values = []
        fscore_values = []
        loss_values = [float(x) for x in re.findall(r'\d+\.\d+e?[-+]?\d*', s[1])]
        for pair in metrics_dict['IoU']:
            iou_values.append(pair[1])

        for pair in metrics_dict['Accuracy']:
            accuracy_values.append(pair[1])
            
        for pair in metrics_dict['F-Score']:
            fscore_values.append(pair[1])

        all_metrics.append({
            'Loss': loss_values,
            'IoU': iou_values,
            'Accuracy': accuracy_values,
            'F-Score': fscore_values,
        })
    return all_metrics