import torch
import numpy as np
from loss_functions.root_tversky_loss import RootTverskyLoss
import warnings
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
    KeepLargestConnectedComponent,
    CenterSpatialCrop
    )
from monai.utils import first, set_determinism
from monai.metrics import DiceMetric, MeanIoU
from monai.inferers import sliding_window_inference
from metrics.precision import Precision
from metrics.recall import Recall
from metrics.tversky_index import TverskyIndex
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR as CosineAnnealingLR
import tqdm
import os


class Netflow:
  def __init__(self, model, device, root_dir, tversky_alpha=0.1, patch_sample_num=16,crop_size = [128, 128, 128], dataloader):
    super().__init__()
    self._model = model

    warnings.filterwarnings("ignore")
    warnings.simplefilter(action='ignore',category=FutureWarning)
    alpha = tversky_alpha
    beta = 1 - alpha
    self.loss_function = RootTverskyLoss(
                                    alpha=alpha,
                                    beta=beta, 
                                    r = 0.7, 
                                    include_background=False,
                                    to_onehot_y=True,
                                    sigmoid=True
                                    )
    self.val_loss_function = RootTverskyLoss(
                                    alpha=alpha,
                                    beta=beta, 
                                    r = 0.7, 
                                    include_background=False,
                                    to_onehot_y=True,
                                    sigmoid=False
                                    )
    
    self.post_pred = Compose([EnsureType(),CenterSpatialCrop([350,350,350]), AsDiscrete(argmax=True, to_onehot=2),
                       KeepLargestConnectedComponent(is_onehot=True,num_components=1)])
    self.post_label = Compose([EnsureType(),CenterSpatialCrop([350,350,350]), AsDiscrete(to_onehot=2)])

    self.dice_metric = DiceMetric(include_background=True, reduction="mean")
    self.precision = Precision(include_background=True, reduction="mean")
    self.recall = Recall(include_background=True, reduction="mean")
    self.iou = MeanIoU(include_background=True, reduction="mean")
    self.tversky_index = TverskyIndex(include_background=True, reduction="mean",
                               alpha=alpha,beta=beta)
    
    self.best_tversky = -1
    self.best_dice = -1
    self.best_precision = -1
    self.best_recall = -1
    self.best_val_epoch = -1
    self.train_ds = None
    self.val_ds = None
    self.test_ds = None
    self.dataloaders = None
    self.prepare_data(test_frac=0.1, val_frac=0.1, patch_sample_num=patch_sample_num, crop_size = crop_size)
    self.root_dir = root_dir
    self.device = device
    
  def load_model(self,model_path):
    self._model.load_state_dict(torch.load(model_path))

  def forward(self, x):
    return self._model(x)
  
  def prepare_data(self, test_frac=0.1, val_frac=0.1, patch_sample_num=16, crop_size = [128, 128, 128]):
    np.random.seed(0)
    set_determinism(seed=0)
    self.dataloaders = dataloader(batch_size_train =2, 
                                  batch_size_val = 1, 
                                  batch_size_test = 1,
                                  sample_num = patch_sample_num,
                                  crop_size = crop_size
                                  )
  
  def configure_optimizers(self):
    scaler = GradScaler()
    optimizer = torch.optim.SGD(self._model.parameters(), lr=1e-2, momentum = 0.9, 
                                weight_decay=0.0001, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, 100, eta_min=1e-4, verbose=True)
    return optimizer, scheduler, scaler
  
  def training_step(self, batch, batch_idx):
    
    images, labels = (
        batch["image"].to(self.device), 
        batch["label"].to(self.device),
        )
    with autocast():
      if self._model.supervision_mode=='decode':
        pred = self.forward(images)
        loss = self.loss_function(pred, labels)
      elif self._model.supervision_mode=='encode_decode':
        pred1, pred2 = self.forward(images)

        loss1 = self.loss_function(pred1, labels)
        loss2 = self.loss_function(pred2, labels)
        loss = (loss1+loss2)/2
      elif self._model.supervision_mode=='deep':
        pred1, pred2, pred3 = self.forward(images)
        loss1 = self.loss_function(pred1, labels)
        loss2 = self.loss_function(pred2, labels)
        loss3 = self.loss_function(pred3, labels)
        loss = (loss1+loss2+loss3)/3
    tensorboard_logs = {"train_loss": loss.item()}
    return {"loss": loss, "log": tensorboard_logs}
  
  def validation_step(self, batch, batch_idx, roi_size = (128, 128, 128), sw_batch_size=1):
    val_images, val_labels = (
        batch["image"].to(self.device), 
        batch["label"].to(self.device),
        )
    pad_len = 16
    val_outputs = sliding_window_inference(val_images, 
                                           roi_size, 
                                           sw_batch_size=sw_batch_size, 
                                           predictor=self.forward)
    border_mask = F.pad(torch.ones(val_images.shape)[:,:,pad_len:-pad_len,pad_len:-pad_len,:], 
                      (0,0, pad_len,pad_len, pad_len,pad_len),
                      value=0).to(self.device)
    val_outputs = val_outputs*border_mask
    val_loss = self.val_loss_function(val_outputs, val_labels)
    val_outputs = [self.post_pred(i) for i in val_outputs]
    val_labels = [self.post_label(i) for i in val_labels]

    self.dice_metric(y_pred=val_outputs, y=val_labels)
    self.precision(y_pred=val_outputs, y=val_labels)
    self.recall(y_pred=val_outputs, y=val_labels)
    self.tversky_index(y_pred=val_outputs, y=val_labels)
    self.iou(y_pred=val_outputs, y=val_labels)

    return {"val_loss": val_loss, "val_len": len(val_outputs)}
  
  def validation_epoch_end(self, outputs, current_epoch):
    val_loss = 0
    num_val_items = 0
    
    # calculate val mean loss and metric
    for output in outputs:
      val_loss += output["val_loss"].sum().item()
      num_val_items += output["val_len"]

    mean_val_dice = self.dice_metric.aggregate().item()
    self.dice_metric.reset()

    mean_precision = self.precision.aggregate().item()
    self.precision.reset()

    mean_recall = self.recall.aggregate().item()
    self.recall.reset()

    mean_tversky_index = self.tversky_index.aggregate().item()
    self.tversky_index.reset()

    mean_iou = self.iou.aggregate().item()
    self.iou.reset()


    mean_val_loss = val_loss / num_val_items
    tensorboard_logs = {
        "DSC": mean_val_dice,
        "Tversky": mean_tversky_index,
        "Precision":mean_precision, 
        "Recall": mean_recall,
        "Val_Loss": mean_val_loss,
        "IoU": mean_iou
    }

    # update best val metric
    if mean_tversky_index > self.best_tversky:
      self.best_tversky = mean_tversky_index
      self.best_dice = mean_val_dice
      self.best_precision = mean_precision
      self.best_recall = mean_recall
      self.best_iou = mean_iou
      self.best_val_epoch = current_epoch + 1
      torch.save(self._model.state_dict(), os.path.join(self.root_dir, "best_metric_model.pth"))
      print("Successfully saved new best model!")
      
    print(
        f"Current epoch: {current_epoch + 1}"
        f"\nCurrent Tversky index: {str(mean_tversky_index)[:6]} | Mean dice: {str(mean_val_dice)[:6]} | Mean IoU: {str(mean_iou)[:6]}"
        f" | Precision: {str(mean_precision)[:6]} | Recall: {str(mean_recall)[:6]}"
        f"\nBest Tversky index: {str(self.best_tversky)[:6]} | Mean dice: {str(self.best_dice)[:6]} | Mean IoU: {str(self.best_iou)[:6]}"
        f" | Precision: {str(self.best_precision)[:6]} | Recall: {str(self.best_recall)[:6]}"
        f" at epoch: {self.best_val_epoch}"
    )
    print("=" * 85)
    return {"log": tensorboard_logs}


class Trainer:
  def __init__(self, writer, max_epochs=100, val_interval=1, roi_size=(128, 128, 128),
               sw_batch_size=1, batch_sizes=(2, 1, 1)):
    self.max_epochs = max_epochs
    self.epoch_loss_values = []
    self.val_interval = val_interval
    self.roi_size = roi_size
    self.sw_batch_size = sw_batch_size
    self.batch_sizes = batch_sizes
    self.writer = writer
    
  

  def fit(self, net, start_epoch=0):
    
    net.train_loader = net.dataloaders.cache_train_patches()
    net.val_loader, net.test_loader = net.dataloaders.cache_val_test_datasets()
    self.optimizer, self.scheduler, self.scaler = net.configure_optimizers()

    if start_epoch > 0:
      net.load_model()
    
    for epoch in range(start_epoch, self.max_epochs):    
      print(f"epoch {epoch + 1}/{self.max_epochs}")
      net._model.train()
      epoch_loss = 0
      train_batch_idx = 0
      
      for train_batch in tqdm(net.train_loader):
        self.optimizer.zero_grad()
        train_batch_idx += 1
        training_step = net.training_step(train_batch, train_batch_idx)
        loss = training_step["loss"]
        log = training_step["log"]

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        epoch_loss += loss.item()
      epoch_loss /= train_batch_idx

      print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
      self.epoch_loss_values.append(epoch_loss)
      self.writer.add_scalar('training loss', epoch_loss, epoch + 1)

      val_batch_idx = 0

      net._model.eval()
      with torch.no_grad():
        val_metric_values = []
        for val_batch in tqdm(net.val_loader):
          val_batch_idx += 1
          val_step = net.validation_step(val_batch, val_batch_idx, 
                                          roi_size = self.roi_size, 
                                          sw_batch_size=self.sw_batch_size)
          
          val_metric_values.append(val_step)
        
        val_end_step = net.validation_epoch_end(val_metric_values, epoch)
        self.scheduler.step()
        self.writer.add_scalar('Tversky Index', 
                          val_end_step["log"]["Tversky"],
                          epoch + 1)
        self.writer.add_scalar('DSC',
                          val_end_step["log"]["DSC"],
                          epoch + 1)
        self.writer.add_scalar('Precision',
                          val_end_step["log"]["Precision"],
                          epoch + 1)
        self.writer.add_scalar('Recall',
                          val_end_step["log"]["Recall"],
                          epoch + 1)
        self.writer.add_scalar('Validation Loss',
                          val_end_step["log"]["Val_Loss"],
                          epoch + 1)
        self.writer.add_scalar('IoU',
                          val_end_step["log"]["IoU"],
                          epoch + 1)
        self.writer.flush()
    
