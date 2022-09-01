import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class LTS_model(pl.LightningModule):
    def __init__(self, use_lr_scheduler = True, lr = 1e-3):
        super().__init__()
        self.use_lr_scheduler = use_lr_scheduler
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat, y

    def configure_optimizers(self):
        '''
        base setting for optimizers and schedulers
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.use_lr_scheduler:
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1, epochs = 100, steps_per_epoch = 36)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            return {'optimizer':optimizer,
                        'lr_scheduler' : scheduler,
                        'monitor':'val_loss'}
        else : 
            return optimizer
    