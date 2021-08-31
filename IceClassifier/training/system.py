import pytorch_lightning as pl
import torch as pt
import torch.nn.functional as F
from torchmetrics import MetricCollection, Accuracy, F1, IoU, Precision, Recall

from IceClassifier.training.models import fetch_backbone_model

optimizer_dict = {
    'SGD': pt.optim.SGD,
    'Adam': pt.optim.Adam,
    'AdamW': pt.optim.AdamW
}


class IceClassifierSystem(pl.LightningModule):
    def __init__(self, num_classes: int, backbone_model: str, optimizer: str, learning_rate: float):
        super(IceClassifierSystem, self).__init__()
        self.save_hyperparameters()

        self.backbone = fetch_backbone_model(backbone_model, num_classes=num_classes)
        self.softmax = pt.nn.LogSoftmax(dim=1)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.__configure_metrics(num_classes)

    def __configure_metrics(self, num_classes):
        metrics_config = {
            "num_classes": num_classes,
            "average": 'micro',
            "mdmc_average": 'global'
        }
        self.train_metrics = MetricCollection(
            [
                Accuracy(**metrics_config),
                F1(**metrics_config)
            ],
            prefix='train/'
        )
        self.val_metrics = MetricCollection(
            [
                Accuracy(**metrics_config),
                Precision(**metrics_config),
                Recall(**metrics_config),
                F1(**metrics_config),
                IoU(num_classes=num_classes)
            ],
            prefix='val/'
        )
        self.test_metrics = self.val_metrics.clone(prefix='test_')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("IceClassifierSystem")
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--backbone_model', type=str, default='deeplabv3_resnet101')
        parser.add_argument('--optimizer', type=str, default='AdamW')
        parser.add_argument('--learning_rate', type=float, default=5e-5)
        return parent_parser

    def forward(self, batch):
        outputs = self.backbone(batch['image'])
        logits = outputs['out']
        probs = self.softmax(logits)
        return probs

    def training_step(self, batch, batch_idx):
        labels = batch['mask']
        logits = self(batch)
        batch_loss = F.nll_loss(logits, labels)

        self.log('train/loss', batch_loss, sync_dist=True)
        return {'loss': batch_loss, 'preds': logits, 'target': labels}

    def training_step_end(self, outputs):
        metrics = self.train_metrics(outputs['preds'], outputs['target'])
        self.log_dict(metrics, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        labels = batch['mask']
        logits = self(batch)
        batch_loss = F.nll_loss(logits, labels)

        self.log('val/loss', batch_loss, sync_dist=True)
        return {'loss': batch_loss, 'preds': logits, 'target': labels}

    def validation_step_end(self, outputs):
        metrics = self.val_metrics(outputs['preds'], outputs['target'])
        self.log_dict(metrics, sync_dist=True)

    def configure_optimizers(self):
        optimizer_class = optimizer_dict[self.optimizer]
        optimizer = optimizer_class(self.parameters(), lr=self.learning_rate)
        return optimizer
