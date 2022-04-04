python train_errnet.py \
    --name errnet_AdaNEC_OF \
    --hyper \
    -r \
    --unaligned_loss vgg \
    --icnn_path ./checkpoints/errnet_ceilnet/errnet_latest.pt \
                ./checkpoints/errnet_unaligned/errnet_latest.pt \
                ./checkpoints/errnet_real90/errnet_latest.pt \
    --nModel 3