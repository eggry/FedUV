import torch
from fedlab.contrib.algorithm.basic_server import SyncServerHandler


class FedUVServerHandler(SyncServerHandler):
    def __init__(
        self,
        model: torch.nn.Module,
        global_round: int,
        num_clients: int,
        sample_ratio: float,
        validate_interval: int,
        cuda: bool = False,
        device: str = None,
        logger=None,
    ):
        super(FedUVServerHandler, self).__init__(model=model, global_round=global_round,
                                                 num_clients=num_clients, sample_ratio=sample_ratio, cuda=cuda, device=device, logger=logger)
        self.validate_interval = validate_interval

    @property
    def if_validate(self):
        return self.round % self.validate_interval == 0
