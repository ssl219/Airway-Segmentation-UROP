from training import Netflow

import torch
import numpy as np
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
    KeepLargestConnectedComponent,
    CenterSpatialCrop
    )
from monai.metrics import DiceMetric, MeanIoU
from monai.inferers import sliding_window_inference
from metrics.precision import Precision
from metrics.recall import Recall
from metrics.tversky_index import TverskyIndex
import tqdm
import os

def test(model, data_dir, device, alpha=0.1, model_path="saved_models/wingsnet_deep.pth"):
  net = Netflow(
    model,
    patch_sample_num=16,
    crop_size = [128, 128, 128])

  val_loader, test_loader = net.dataloaders.cache_val_test_datasets()
  net.load_model(model_path=os.path.join(data_dir, model_path))
  model = net._model
  model.eval()
  post_pred = Compose([EnsureType(),CenterSpatialCrop([350,350,350]), AsDiscrete(argmax=True, to_onehot=2),
                       KeepLargestConnectedComponent(is_onehot=True,num_components=1)])
  post_label = Compose([EnsureType(),CenterSpatialCrop([350,350,350]), AsDiscrete(to_onehot=2)])
  
  # define metrics
  dice_metric = DiceMetric(include_background=True, reduction="mean")
  precision = Precision(include_background=True, reduction="mean")
  recall = Recall(include_background=True, reduction="mean")
  iou = MeanIoU(include_background=True, reduction="mean")
  tversky_index = TverskyIndex(include_background=True, reduction="mean",
                               alpha=alpha,beta=1-alpha)


  with torch.no_grad():

    for test_data in tqdm(test_loader):
        test_inputs, test_labels = (
          test_data["image"].to(device),
          test_data["label"].to(device)
          )

        pad_len=16

        border_mask = F.pad(torch.ones(test_inputs.shape)[:,:,pad_len:-pad_len,pad_len:-pad_len,:], 
                           (0,0, pad_len,pad_len, pad_len,pad_len),
                           value=0).to(device)

        roi_size = (128, 128, 128) 
        test_outputs = sliding_window_inference(test_inputs, 
                                               roi_size, 
                                               sw_batch_size=1, 
                                               predictor=model)
        test_outputs = test_outputs*border_mask
        test_labels = test_labels*border_mask

        test_outputs = [post_pred(i) for i in test_outputs]  # decollate_batch(val_outputs)
        test_labels = [post_label(i) for i in test_labels]  #  decollate_batch(val_labels)]


        # compute metric for current iteration
        dice_metric(y_pred=test_outputs, y=test_labels)
        precision(y_pred=test_outputs, y=test_labels)
        recall(y_pred=test_outputs, y=test_labels)
        tversky_index(y_pred=test_outputs, y=test_labels)
        iou(y_pred=test_outputs, y=test_labels)

    # aggregate the final mean dice result
    dsc_metric = dice_metric.aggregate().item()
    precision_metric = precision.aggregate().item()
    recall_metric = recall.aggregate().item()
    tversky_index_metric = tversky_index.aggregate().item()
    iou_metric = iou.aggregate().item()

    # reset the status for next round
    dice_metric.reset()
    precision.reset()
    recall.reset()
    tversky_index.reset()
    iou.reset()

    print('Dice Score Coeficient: ' + str(dsc_metric))
    print()
    print('Precision: ' + str(precision_metric))
    print()
    print('Recall: ' + str(recall_metric))
    print()
    print('Tversky Index: ' + str(tversky_index_metric))
    print()
    print('IoU: ' + str(iou_metric))
    """
    print('=========================================================================')
    print('Example Scan:')
    print('=========================================================================')
    save_itk(test_outputs[0][1].cpu().detach().numpy(), [0,0,0,0], [1/128,1/128,1/128,1/128], os.path.join('UROP_2022/airway_segmentation/test_output/', '_test_output_'+'.nii.gz'))
    save_itk(test_labels[0][1].cpu().detach().numpy(), [0,0,0,0], [1/128,1/128,1/128,1/128], os.path.join('UROP_2022/airway_segmentation/test_output/', '_test_label_'+'.nii.gz'))

    # display last batch's results
    for i in range(0,test_labels[0].shape[-1],20):
      f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
      ax1.set_title('Scan')

      ax1.imshow(test_inputs[0][0,:,:,i].cpu().detach().numpy())

      ax2.set_title('Output')
      ax2.imshow(test_outputs[0][1,:,:,i].cpu().detach().numpy())

      ax3.set_title('Label')
      ax3.imshow(test_labels[0][1,:,:,i].cpu().detach().numpy())

      ax4.set_title('Mask')
      ax4.imshow(border_mask[0][0,:,:,i].cpu().detach().numpy())
      plt.show()


      
      print('=========================================================================')
      """
   
  return 


