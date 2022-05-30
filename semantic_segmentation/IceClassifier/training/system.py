from typing import List

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
    def __init__(
            self,
            num_classes: int,
            backbone_model: str,
            optimizer: str,
            learning_rate: float,
            loss_weights: List[float]
    ):
        super(IceClassifierSystem, self).__init__()
        self.save_hyperparameters()

        self.backbone = fetch_backbone_model(backbone_model, num_classes=num_classes)
        self.softmax = pt.nn.LogSoftmax(dim=1)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.__configure_metrics(num_classes)
        self.loss_weights = pt.FloatTensor(loss_weights)

    def __configure_metrics(self, num_classes):
        metrics_config = {
            "num_classes": num_classes,
            "mdmc_average": 'global'
        }
        self.train_metrics = [
            MetricCollection(
                [
                    Accuracy(**metrics_config, average=average),
                    F1(**metrics_config, average=average),
                ],
                prefix='train/',
                postfix=f'_{average}'
            )
            for average in ['micro', 'macro']
        ]
        self.val_metrics = [
            MetricCollection(
                [
                    Accuracy(**metrics_config, average=average),
                    Precision(**metrics_config, average=average),
                    Recall(**metrics_config, average=average),
                    F1(**metrics_config, average=average),
                ],
                prefix='val/',
                postfix=f'_{average}'
            )
            for average in ['micro', 'macro']
        ]
        self.val_metrics.append(
            MetricCollection(
                [
                    IoU(num_classes=num_classes)
                ],
                prefix='val/'
            )
        )
        self.test_metrics = [mc.clone(prefix='test/') for mc in self.val_metrics]

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("IceClassifierSystem")
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--backbone_model', type=str, default='deeplabv3_resnet101')
        parser.add_argument('--dataset', type=str, default='original')
        parser.add_argument('--local_mode', type=bool, default=False)
        parser.add_argument('--dataset_path', type=str,
                            default='./resources/datasets/ice-tiles-dataset-badlight/256x256')
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
        loss_weights = self.loss_weights.to(device=self.device)
        batch_loss = F.nll_loss(logits, labels, weight=loss_weights)

        self.log('train/loss', batch_loss, sync_dist=True)
        return {'loss': batch_loss, 'preds': logits.detach(), 'target': labels.detach()}

    def training_step_end(self, outputs):
        for train_metrics in self.train_metrics:
            train_metrics = train_metrics.to(self.device)
            metrics = train_metrics(outputs['preds'], outputs['target'])
            self.log_dict(metrics, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        labels = batch['mask']
        logits = self(batch)
        loss_weights = self.loss_weights.to(device=self.device)
        batch_loss = F.nll_loss(logits, labels, weight=loss_weights)

        self.log('val/loss', batch_loss, sync_dist=True)
        return {'loss': batch_loss, 'preds': logits.detach(), 'target': labels.detach()}

    def validation_step_end(self, outputs):
        for val_metrics in self.val_metrics:
            val_metrics = val_metrics.to(self.device)
            metrics = val_metrics(outputs['preds'], outputs['target'])
            self.log_dict(metrics, sync_dist=True)

    def test_step(self, batch, batch_idx):
        labels = batch['mask']
        logits = self(batch)
        loss_weights = self.loss_weights.to(device=self.device)
        batch_loss = F.nll_loss(logits, labels, weight=loss_weights)

        self.log('test/loss', batch_loss, sync_dist=True)
        return {'loss': batch_loss, 'preds': logits.detach(), 'target': labels.detach()}

    def test_step_end(self, outputs):
        for test_metrics in self.test_metrics:
            test_metrics = test_metrics.to(self.device)
            metrics = test_metrics(outputs['preds'], outputs['target'])
            self.log_dict(metrics, sync_dist=True)

    def configure_optimizers(self):
        optimizer_class = optimizer_dict[self.optimizer]
        optimizer = optimizer_class(self.parameters(), lr=self.learning_rate)
        return optimizer
