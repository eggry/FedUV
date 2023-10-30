
from copy import deepcopy
import torch

from fedlab.core.client.trainer import SerialClientTrainer
from fedlab.utils.functional import AverageMeter


class FedUVSerialClientTrainer(SerialClientTrainer):
    def __init__(self, model, num_clients, cuda=False, device=None, personal=False) -> None:
        super().__init__(model=model, num_clients=num_clients,
                         cuda=cuda, device=device, personal=personal)
        self.trained_models = []
        self.loss_ = AverageMeter()

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr, lr_decay, lr_decay_step_size):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_step_size = lr_decay_step_size
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, gamma=lr_decay, step_size=lr_decay_step_size)

    @property
    def uplink_package(self):
        trained_models = deepcopy(self.trained_models)
        loss = self.loss_.avg
        lr = self.scheduler.get_last_lr()
        self.trained_models = []
        self.loss_.reset()
        self.scheduler.step()
        return lr, loss, trained_models

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for cid in id_list:
            dataloader = self.dataset.get_dataloader(
                cid, "train_pos", self.batch_size, shuffle=True)
            user_code = self.dataset.get_user_code(cid)
            trained_model, loss_avg, loss_n = self.train(
                model_parameters, dataloader, user_code)
            self.trained_models.append(trained_model)
            self.loss_.update(loss_avg, loss_n)

    def local_validate_process(self, payload, id_list):
        model_parameters = payload[0]
        self.set_model(model_parameters)

        user_tau_ = AverageMeter()
        pos_e_ = AverageMeter()
        tpr_ = AverageMeter()
        pos_loss_ = AverageMeter()
        neg_e_ = AverageMeter()
        fpr_ = AverageMeter()

        for cid in id_list:
            user_code = self.dataset.get_user_code(cid)

            # use train dataset for warmup
            warmup_loader = self.dataset.get_dataloader(cid, 'train_pos')

            user_tau = self.warmup(warmup_loader, user_code, 0.9)
            user_tau_.update(user_tau, 1)

            pos_loader = self.dataset.get_dataloader(cid, 'val_pos', 256)
            e, accept, loss, count = self.eval(pos_loader, user_code, user_tau)
            pos_e_.update(e, count)
            tpr_.update(accept, count)
            pos_loss_.update(loss, count)

            neg_loader = self.dataset.get_dataloader(cid, 'val_neg', 256)
            e, accept, loss, count = self.eval(neg_loader, user_code, user_tau)
            neg_e_.update(e, count)
            fpr_.update(accept, count)
        return {
            "val_tpr": tpr_.avg,
            "val_fpr": fpr_.avg,
            "val_TP": tpr_.sum,
            "val_FP": fpr_.sum,
            "val_TN": fpr_.count-fpr_.sum,
            "val_FN": tpr_.count-tpr_.sum,
            "val_pos_loss": pos_loss_.avg,
            "val_user_tau": user_tau_.avg,
            "val_pos_e": pos_e_.avg,
            "val_neg_e": neg_e_.avg,
        }

    def local_evaluate_process(self, payload, id_list):
        model_parameters = payload[0]
        self.set_model(model_parameters)

        user_tau_ = AverageMeter()
        pos_e_ = AverageMeter()
        tpr_ = AverageMeter()
        pos_loss_ = AverageMeter()
        neg_e_ = AverageMeter()
        fpr_ = AverageMeter()
        extra_neg_e_ = AverageMeter()
        extra_fpr_ = AverageMeter()

        for cid in id_list:
            user_code = self.dataset.get_user_code(cid)

            # use train dataset for warmup
            warmup_loader = self.dataset.get_dataloader(cid, 'train_pos')

            user_tau = self.warmup(warmup_loader, user_code, 0.9)
            user_tau_.update(user_tau, 1)

            pos_loader = self.dataset.get_dataloader(cid, 'test_pos', 256)
            e, accept, loss, count = self.eval(pos_loader, user_code, user_tau)
            pos_e_.update(e, count)
            tpr_.update(accept, count)
            pos_loss_.update(loss, count)

            neg_loader = self.dataset.get_dataloader(cid, 'test_neg', 256)
            e, accept, loss, count = self.eval(neg_loader, user_code, user_tau)
            neg_e_.update(e, count)
            fpr_.update(accept, count)

            extra_neg_loader = self.dataset.get_dataloader(
                cid, 'extra_test_neg', 256)
            e, accept, loss, count = self.eval(
                extra_neg_loader, user_code, user_tau)
            extra_neg_e_.update(e, count)
            extra_fpr_.update(accept, count)
        return {
            "test_tpr": tpr_.avg,
            "test_fpr": fpr_.avg,
            "test_TP": tpr_.sum,
            "test_FP": fpr_.sum,
            "test_extra_FP": extra_fpr_.sum,
            "test_TN": fpr_.count-fpr_.sum,
            "test_extra_TN": extra_fpr_.count-extra_fpr_.sum,
            "test_FN": tpr_.count-tpr_.sum,
            "test_pos_loss": pos_loss_.avg,
            "test_user_tau": user_tau_.avg,
            "test_pos_e": pos_e_.avg,
            "test_neg_e": neg_e_.avg,
            "test_extra_neg_e": extra_neg_e_.avg,
        }

    def train(self, model_parameters, train_loader, user_code):
        self.set_model(model_parameters)
        self._model.train()
        if self.cuda:
            user_code = user_code.cuda(self.device)
        loss_ = AverageMeter()
        for _ in range(self.epochs):
            for data, label in train_loader:
                batch_size = len(label)
                if self.cuda:
                    data = data.cuda(self.device)
                self.optimizer.zero_grad()
                embeddings = self._model(data)
                e = embeddings.mv(user_code)/self._model.c
                loss = (1-e).clamp_min(0).mean()  # l_pos
                loss.backward()
                self.optimizer.step()
                loss_.update(loss.item(), batch_size)
        return [self.model_parameters], loss_.avg, loss_.count

    def warmup(self, warmup_loader, user_code, q=0.9):
        self.model.eval()
        if self.cuda:
            user_code = user_code.cuda(self.device)
        e = torch.empty((0), dtype=torch.float32)
        if self.cuda:
            e = e.cuda(self.device)
        for data, _ in warmup_loader:
            if self.cuda:
                data = data.cuda(self.device)
            embeddings = self.model(data)
            e = torch.cat((e, embeddings.mv(user_code)/self.model.c))
        return e.quantile(1-q).item()

    def eval(self, dataloader, user_code, user_tau):
        self.model.eval()
        if self.cuda:
            user_code = user_code.cuda(self.device)
        e_ = AverageMeter()
        accept_ = AverageMeter()
        loss_ = AverageMeter()
        for data, labels in dataloader:
            batch_size = len(labels)
            if self.cuda:
                data = data.cuda(self.device)
            embeddings = self.model(data)
            e = embeddings.mv(user_code)/self.model.c
            accept = (e >= user_tau)
            loss = (1-e).clamp(min=0).mean()
            e_.update(e.mean().item(), batch_size)
            accept_.update(accept.float().mean().item(), batch_size)
            loss_.update(loss.item(), batch_size)
        return e_.avg, accept_.avg, loss_.avg, loss_.count
