import torchio as tio
import torch

def get_training_transform():
    return tio.Compose([
            tio.ToCanonical(),
            tio.RandomMotion(p=0.2),
            tio.RandomBiasField(p=0.3),
            tio.RandomNoise(p=0.5),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean),
            tio.RandomAffine()
        ])

def get_validation_transform():
    return tio.Compose([
            tio.ToCanonical(),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean)
        ])

def get_train_val_sets(image_paths):

    subjects = []
    for image_path in image_paths:
        subject = tio.Subject(
            mri=tio.ScalarImage(image_path),
        )
        subjects.append(subject)

    training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
    num_subjects = len(subjects)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    training_subjects, validation_subjects = torch.utils.data.random_split(subjects, num_split_subjects)

    training_set = tio.SubjectsDataset(
        training_subjects, transform=get_training_transform())

    validation_set = tio.SubjectsDataset(
        validation_subjects, transform=get_validation_transform())
    
    return training_set, validation_set

def get_pretrain_set(label_paths):

    subjects = []
    for label_path in label_paths:
        subject = tio.Subject(
            mri=tio.ScalarImage(label_path.__str__().replace('labels-resampled', 'images-resampled')),
            labels=tio.LabelMap(label_path)
        )
        subjects.append(subject)

    training_set = tio.SubjectsDataset(subjects, transform=get_validation_transform())

    return training_set
