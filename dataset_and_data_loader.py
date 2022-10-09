from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader

# CUB200 2011 Dataset

# Data transform
transform_train = T.Compose([T.RandomResizedCrop(224),
                             T.RandomHorizontalFlip(),
                             T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], 
                                           p=0.2),
                             T.RandomGrayscale(0.5),
                             T.ToTensor(),
                             T.Normalize((0.4914, 0.4822, 0.4465), 
                                         (0.2023, 0.1994, 0.2010))])

transform_test = T.Compose([T.Resize((224, 224)),
                            T.ToTensor(),
                            T.Normalize((0.4914, 0.4822, 0.4465), 
                                        (0.2023, 0.1994, 0.2010))])


# DataLoader
train_set = datasets.ImageFolder(root='./CUB_200_2011/train/', transform=transform_train)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2)

test_set = datasets.ImageFolder(root='./CUB_200_2011/test/', transform=transform_test)
test_loader = DataLoader(test_set, batch_size=8, shuffle=True, num_workers=2)


