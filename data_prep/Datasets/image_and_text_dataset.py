import torch
from torchvision.datasets import ImageFolder
import csv
import random

VECTOR_LEN = 15
STDEV = 5.066
PROB = 0.95


class ImageAndTextDataset(ImageFolder):
    """
    A dataset for retrieving images with text vectors
    """

    def __init__(self, path, transforms, target_transforms, vector_csv_path, mode="train"):
        """
        __init__ is run once upon initializing the Dataset object
        :param vector_csv_path: Path to a csv file representing a list of contextual vectors
        where the index is the label
        label | context vector of label
        context_vectors: a dict based on the vector_csv_path:
            { label : vector of career } for each career
        """
        super().__init__(path, transforms, target_transform=target_transforms)
        self.context_vectors = dict()
        with open(vector_csv_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                context_vector_label = row[0]
                if context_vector_label == "":
                    continue
                self.context_vectors[context_vector_label] = torch.Tensor(
                    [float(i) for i in row[1:1+VECTOR_LEN]])
        self.mode = "train"

    def __getitem__(self, idx):
        """
        Loads a sample from the dataset at the given :param: idx
        :return: the loaded sample (image, label, contextual vector)
        """
        path, label = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        context_vector_label = path.split("/")[-2]
        vector = self.context_vectors[context_vector_label]

        if self.mode == "train":
            rand_prob = random.random()
            # if random probability < PROB the context vector is the original
            # else the context vector is rendom
            if rand_prob < PROB:
                context_vector_label = path.split("/")[-2]
            else:
                context_vector_label = random.choice(list(self.context_vectors))

        if self.mode == "train" or self.mode == "test":
            # add randomness to the context vector
            vector = self.context_vectors[context_vector_label] + torch.normal(mean=0.0, std=STDEV, size=self.context_vectors[context_vector_label].size())
        
        return image, label, vector
