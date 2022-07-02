from email import header
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from data_prep.Datasets.image_and_text_dataset import ImageAndTextDataset
from modelling.models.context_vgg16 import context_vgg16
from sklearn.metrics import classification_report
import pandas as pd

PATH_IMAGE_FOLDER = "/home/ssd_storage/datasets/processed/context_vggfaces_num-classes_1050_{'train': 0.7, 'val': 0.2, 'test': 0.1}/test"
PATH_CSV = "/home/context/facial_feature_impact_comparison/extracted_data/sanity_check.csv"
PATH_MODEL = "/home/ssd_storage/experiments/students/context/context_vgg16/context_vgg16/models/best.pth"
PATH_RESULTS = "/home/context/results.csv"
BATCH_SIZE=32
WORKERS=4
NUM_CLASSES = 1050
OPTIMIZER_PARAMS = {
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4}

def testing():
    transform = transforms.Compose(
    [transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = ImageAndTextDataset(path=PATH_IMAGE_FOLDER,
                                transforms=transform,
                                target_transforms=None,
                                vector_csv_path=PATH_CSV)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=False)
    model = context_vgg16(False, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), **OPTIMIZER_PARAMS)
    # lr_scheduler
    model_checkpoint = torch.load(PATH_MODEL)
    model, optimizer, acc_train, epoch_train = load_model_and_optimizer_loc(model, optimizer, PATH_MODEL)
    
    y_pred_list = []
    y_actual_list = []
    num_correct = 0
    sum_loss = 0
    with torch.no_grad():
        model.eval()
        for images, labels, context_vectors in dataloader:
            images = images.to(device)
            context_vectors = context_vectors.to(device)
            labels = labels.to(device)

            y_test_pred = model(images, context_vectors)
            sum_loss += criterion(y_test_pred, labels)
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            num_correct += sum(labels == y_pred_tags).item()

            y_actual_list.extend(labels.tolist())
            y_pred_list.extend(y_pred_tags.tolist())
    
    df = pd.DataFrame(list(zip(y_actual_list, y_pred_list)), columns=['actual class', 'predicted class'])
    df.to_csv(PATH_RESULTS, index=False)

    print(num_correct)
    print(classification_report(y_actual_list, y_pred_list))
    print("loss:" + str(sum_loss.item()/len(y_actual_list)))
    print("accuracy:" + str(num_correct/len(y_actual_list)))

def load_model_and_optimizer_loc(model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, model_location=None):
    with open(model_location, 'br') as f:
        print("Loading model from: ", model_location)
        model_checkpoint = torch.load(f)
        state_dict_updated = {key[7:] : value for key, value in model_checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict_updated)
        if optimizer is not None:
            optimizer.load_state_dict(model_checkpoint['optimizer'])
    print(f"loading weights from {model_location}, acc={model_checkpoint['acc']}, epoch={model_checkpoint['epoch']}")
    return model, optimizer, model_checkpoint['acc'], model_checkpoint['epoch']
    

if __name__ == '__main__':
    testing()