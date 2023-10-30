import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets, transforms


def group_indices_by_celeba_id(dataset):
    indices_by_cecleba_id = {}

    for index, celeba_id in enumerate(dataset.identity.flatten().tolist()):
        indices = indices_by_cecleba_id.get(celeba_id, [])
        indices.append(index)
        indices_by_cecleba_id[celeba_id] = indices
    return indices_by_cecleba_id


def sample_users(dataset, num_users):
    indices_by_celeba_id = group_indices_by_celeba_id(dataset)

    # sample 30 images for each user
    users = [
        {
            "id": celeba_id,
            "secret": random.getrandbits(64),
            "indices": random.sample(indices, 30)
        }
        for celeba_id, indices in indices_by_celeba_id.items()
        if len(indices) >= 30
    ]
    users = random.sample(users, num_users)
    return users


def sample_extra(dataset, num_extra):
    indices_by_celeba_id = group_indices_by_celeba_id(dataset)

    # sample the separate test set
    return random.sample([
        random.sample(indices, 1)[0]
        for _, indices in indices_by_celeba_id.items()
    ], num_extra)


def prepare_all_users_sample(users, extra_test):
    all_train = set()
    all_val = set()
    all_test = set()

    for user in users:
        user |= {
            "train_pos": user["indices"][:20],
            "val_pos": user["indices"][20:25],
            "test_pos": user["indices"][25:30],
        }
        all_train |= set(user["train_pos"])
        all_val |= set(user["val_pos"])
        all_test |= set(user["test_pos"])

    for user in users:
        # `train_neg` is not used for FedUV
        user |= {
            "train_neg": list(all_train-set(user["train_pos"])),
            "val_neg": list(all_val-set(user["val_pos"])),
            "test_neg": list(all_test-set(user["test_pos"])),
            "extra_test_neg": extra_test,
        }


def generate_user_code(bch, user):
    b_u = np.binary_repr(user["id"], 32)  # l_b = 32
    r_u = np.binary_repr(user["secret"], 64)
    m_u = [int(x) for x in b_u+r_u]
    assert (len(m_u) >= bch.k)
    v_u = bch.encode(m_u[:bch.k])
    assert (len(v_u) == bch.n)
    return torch.tensor([-1 if x == 0 else 1 for x in v_u], dtype=torch.float32)


def celeba_default_transforms(normalize):
    return (
        transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )
        if normalize
        else transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )
    )


class PartitionedCelebA():
    def __init__(self,
                 root,
                 num_clients=1000,
                 num_extra=1000,
                 download=False,
                 seed=None,
                 transforms=None,
                 normalize=True,
                 bch=None) -> None:
        self.root = root
        self.num_clients = num_clients
        self.num_extra = num_extra
        self.normalize = normalize
        self.transforms = transforms or celeba_default_transforms(normalize)
        self.download = download
        self.seed = seed
        self.bch = bch

    def prepare(self):
        # for clients
        self.celeba_train = datasets.CelebA(self.root, split='train',
                                            download=self.download, transform=self.transforms, target_type="identity")

        # for extra tests
        self.celeba_test = datasets.CelebA(self.root, split='test',
                                           download=self.download, transform=self.transforms, target_type="identity")

        np.random.seed(self.seed)
        self.users = sample_users(self.celeba_train, self.num_clients)
        self.extra = sample_extra(self.celeba_test, self.num_extra)
        prepare_all_users_sample(self.users, self.extra)
        self.user_codes = [generate_user_code(
            self.bch, user) for user in self.users]

    def get_dataset(self, cid, type):
        dataset = self.celeba_test if type.startswith(
            "extra_") else self.celeba_train
        return Subset(dataset, self.users[cid][type])

    def get_dataloader(self, cid, type, batch_size=None, shuffle=False):
        dataset = self.get_dataset(cid, type)
        batch_size = batch_size or len(dataset)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_user_code(self, cid):
        return self.user_codes[cid]
