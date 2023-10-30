from feduv.handler import FedUVServerHandler
from feduv.trainer import FedUVSerialClientTrainer
from tensorboardX import SummaryWriter


class FedUVPipeline():
    def __init__(self, handler: FedUVServerHandler, trainer: FedUVSerialClientTrainer, metric_writter: SummaryWriter, verbose=False):
        self.handler = handler
        self.trainer = trainer
        self.metric_logger = metric_writter
        self.verbose = verbose

    def main(self):
        sampled_eval_clients = self.handler.sample_clients()
        while self.handler.if_stop is False:
            r = self.handler.round
            if self.verbose:
                print(f"\rTraining at round {r}", end="")

            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package

            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            lr, loss, trained_models = self.trainer.uplink_package

            # server side
            self.metric_logger.add_scalar('lr', lr, r)
            self.metric_logger.add_scalar('train_loss', loss, r)
            for trained_model in trained_models:
                self.handler.load(trained_model)

            if self.handler.if_validate:
                if self.verbose:
                    print(f"\rValidating at round {r}", end="")
                broadcast = self.handler.downlink_package
                result = self.trainer.local_validate_process(
                    broadcast, sampled_eval_clients)
                for k, v in result.items():
                    self.metric_logger.add_scalar(k, v, r)
                if self.verbose:
                    print(f"\rAfter Round {r}, val tesult: {result}")

        r = self.handler.round-1

        if self.verbose:
            print(f"\rTesting at round {r}", end="")

        broadcast = self.handler.downlink_package
        result = self.trainer.local_evaluate_process(
            broadcast, sampled_eval_clients)
        for k, v in result.items():
            self.metric_logger.add_scalar(k, v, r)
        if self.verbose:
            print(f"\rAfter Round {r}, test result: {result}")
