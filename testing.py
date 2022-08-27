import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_prep.Datasets.image_and_text_dataset import ImageAndTextDataset
from modelling.models.context_vgg16 import context_vgg16
import pandas as pd
from scipy.stats import entropy

PATH_IMAGE_FOLDER = "/home/ssd_storage/datasets/processed/context_vggfaces_num-classes_1050_{'train': 0.7, 'val': 0.2, 'test': 0.1}/test"
PATH_CSV_TRAIN = "/home/context/facial_feature_impact_comparison/extracted_data/occupations_vectors/train.csv"
PATH_MODEL = "/home/ssd_storage/experiments/students/context/context_vgg16_5/context_vgg16/models/best.pth"
PATH_RESULTS = "/home/context/results.csv"
BATCH_SIZE = 128
WORKERS = 4
NUM_CLASSES = 1050


def testing(path_image_train=PATH_IMAGE_FOLDER, path_image_test=PATH_IMAGE_FOLDER, path_csv_train=PATH_CSV_TRAIN,
            path_csv_test=PATH_CSV_TRAIN,
            path_model=PATH_MODEL, path_results=PATH_RESULTS, which=1, mode="test"):
    """
    Runs the relevant test and saves all the results and the calculated metrics that achieved in this test
    """
    transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_test = ImageAndTextDataset(path=path_image_test,
                                       transforms=transform,
                                       target_transforms=None,
                                       vector_csv_path=path_csv_test, mode=mode)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()
    dataloader = DataLoader(
        dataset_test,
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
        shuffle=True,
        pin_memory=pin_memory,
        drop_last=False)
    model = context_vgg16(False, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    model = load_model_and_optimizer_loc(model, path_model)

    label_to_context_vector_label_dict_test = {label: path.split("/")[-2] for path, label in dataset_test.samples}

    dataset_train = ImageAndTextDataset(path=path_image_train,
                                        transforms=transform,
                                        target_transforms=None,
                                        vector_csv_path=path_csv_test, mode=mode)
    label_to_context_vector_label_dict_train = {label: path.split("/")[-2] for path, label in dataset_train.samples}

    df_test = pd.read_csv(path_csv_test)
    get_SOC_code_test = lambda label: \
        df_test[df_test.iloc[:, 0] == label_to_context_vector_label_dict_test[label]]["O*NET-SOC Code"].iloc[0]

    df_train = pd.read_csv(path_csv_train)
    get_SOC_code_train = lambda label: \
        df_train[df_train.iloc[:, 0] == label_to_context_vector_label_dict_train[label]]["O*NET-SOC Code"].iloc[0]

    y_pred_list = []
    y_actual_list = []
    y_pred_entropy_list = []
    y_pred_scores_list = []
    test3_list = []
    test2_list = []
    paths_list = []
    num_correct = 0
    num_correct_test = 0
    sum_loss = 0
    with torch.no_grad():
        model.eval()
        for images, labels, context_vectors, paths in dataloader:
            images = images.to(device)
            context_vectors = context_vectors.to(device)
            labels = labels.to(device)

            y_test_pred = model(images, context_vectors)
            sum_loss += criterion(y_test_pred, labels)
            y_pred_scores, y_pred_tags = torch.max(y_test_pred, dim=1)
            num_correct += sum(labels == y_pred_tags).item()

            y_actual_list.extend(labels.tolist())
            y_pred_list.extend(y_pred_tags.tolist())
            y_pred_scores_list.extend(y_pred_scores.tolist())
            paths_list.extend(paths)

            y_pred_softmax = torch.softmax(y_test_pred, dim=1).cpu()
            y_pred_entropy_list.extend(entropy(y_pred_softmax, axis=1))

            if which == 3:
                labels_SOC_code = labels.to(device="cpu", dtype=torch.float64).apply_(get_SOC_code_test)
                y_pred_tags_SOC_code = y_pred_tags.to(device="cpu", dtype=torch.float64).apply_(get_SOC_code_train)
                correct_SOC_code = labels_SOC_code == y_pred_tags_SOC_code
                num_correct_test += sum(correct_SOC_code).item()
                test3_list.extend(correct_SOC_code.tolist())
                df = pd.DataFrame(list(
                    zip(paths_list, y_actual_list, y_pred_list, y_pred_entropy_list, y_pred_scores_list, test3_list)),
                                  columns=['path', 'actual class', 'predicted class', 'entropy',
                                           'predicted class score', 'correct based SOC code'])
                df.to_csv(path_results, index=False)

            if which == 2:
                labels_SOC_code = labels.to(device="cpu", dtype=torch.float64).apply_(get_SOC_code_test)
                y_pred_tags_SOC_code = y_pred_tags.to(device="cpu", dtype=torch.float64).apply_(get_SOC_code_train)
                errors_based_contest = [
                    1 if labels_SOC_code[i] == y_pred_tags_SOC_code[i] and labels[i] != y_pred_tags[i] else 0 for i in
                    range(len(labels))]
                num_correct_test += sum(errors_based_contest)
                test2_list.extend(errors_based_contest)
                df = pd.DataFrame(list(
                    zip(paths_list, y_actual_list, y_pred_list, y_pred_entropy_list, y_pred_scores_list, test2_list)),
                                  columns=['path', 'actual class', 'predicted class', 'entropy',
                                           'predicted class score', 'errors based contest'])
                df.to_csv(path_results, index=False)

    if which != 3 and which != 2:
        df = pd.DataFrame(list(zip(paths_list, y_actual_list, y_pred_list, y_pred_entropy_list, y_pred_scores_list)),
                          columns=['path', 'actual class', 'predicted class', 'entropy', 'predicted class score'])
        df.to_csv(path_results, index=False)

    print("saved to: ", path_results)
    print("loss: " + str(sum_loss.item() / len(y_actual_list)))
    print("accuracy: " + str(num_correct / len(y_actual_list)))
    if which == 2:
        print("percent errors based on context: " + str(num_correct_test / (len(y_actual_list) - num_correct)))
    if which == 3:
        print("accuracy based SOC code: " + str(num_correct_test / len(y_actual_list)))


def load_model_and_optimizer_loc(model: torch.nn.Module, model_location=None):
    """
    Load the trained model
    """
    with open(model_location, 'br') as f:
        print("Loading model from: ", model_location)
        model_checkpoint = torch.load(f)
        state_dict_updated = {key[7:]: value for key, value in model_checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict_updated)
    return model


def run_test(which=1, path_csv_test=PATH_CSV_TRAIN):
    """
    Runs the relevant test with the appropriate arguments based on the test type
    """
    path_image_folder_common = "/home/ssd_storage/datasets/processed/context_vggfaces_num-classes_1050_{'train': 0.7, 'val': 0.2, 'test': 0.1}/test"
    path_image_folder_not_common = "/home/context/dataset/test_unfamiliar"
    path_results = "/home/context/facial_feature_impact_comparison/test_results_data/results_" + path_csv_test.split("/")[-1]
    if which == 1:
        testing(path_image_test=path_image_folder_common, path_csv_train=PATH_CSV_TRAIN, path_csv_test=path_csv_test,
                path_results=path_results, which=1)
    elif which == 2:
        testing(path_image_test=path_image_folder_common, path_csv_train=PATH_CSV_TRAIN, path_csv_test=path_csv_test,
                path_results=path_results, which=2)
    elif which == 3:
        testing(path_image_test=path_image_folder_not_common, path_csv_train=PATH_CSV_TRAIN,
                path_csv_test=path_csv_test,
                path_results=path_results, which=3)


def run_all_our_test():
    """
    Runs all the test for a trained model
    """
    print("test 1:")
    run_test(which=1, path_csv_test=PATH_CSV_TRAIN)

    print("test 2:")
    run_test(which=2,
             path_csv_test="/home/context/facial_feature_impact_comparison/extracted_data/occupations_vectors/test2.csv")

    print("test 2_1:")
    run_test(which=2, path_csv_test="/home/context/facial_feature_impact_comparison/extracted_data/occupations_vectors/test2_1.csv")

    print("test 2_2:")
    run_test(which=2,
             path_csv_test="/home/context/facial_feature_impact_comparison/extracted_data/occupations_vectors/test2_2.csv")

    print("test 2_3:")
    run_test(which=2,
             path_csv_test="/home/context/facial_feature_impact_comparison/extracted_data/occupations_vectors/test2_3.csv")

    print("test 3:")
    run_test(which=3,
             path_csv_test="/home/context/facial_feature_impact_comparison/extracted_data/occupations_vectors/test3.csv")

    print("test 3_1:")
    run_test(which=3, path_csv_test="/home/context/facial_feature_impact_comparison/extracted_data/occupations_vectors/test3_1.csv")


def run_test_sanity_check():
    """
    Runs a sanity check test with zero context vectors
    """
    path_sanity_check = "/home/context/facial_feature_impact_comparison/extracted_data/occupations_vectors/sanity_check.csv"
    path_model = "/home/ssd_storage/experiments/students/context/context_vgg16/context_vgg16/models/best.pth"
    path_results = "/home/context/facial_feature_impact_comparison/test_results_data/results_sanity_check.csv"

    print("sanity_check:")
    testing(path_image_test=PATH_IMAGE_FOLDER, path_csv_train=path_sanity_check, path_csv_test=path_sanity_check,
            path_model=path_model, path_results=path_results, which=1, mode="sanity_check")


if __name__ == '__main__':
    run_test_sanity_check()
    run_all_our_test()
