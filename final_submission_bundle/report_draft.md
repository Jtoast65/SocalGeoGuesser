# Human Baseline

Add your human accuracy from the SoCalGuessr game here.

Add your screenshot of the confusion matrix here.

Add a short note about how well you performed and what visual cues you found useful.

**Final Model Architecture**

My final model is a convolutional neural network based on MobileNetV3-Small with transfer learning. I used the ImageNet-pretrained MobileNetV3-Small backbone from `torchvision` and replaced the final classifier so that the network predicts one of the six SoCalGuessr classes: Anaheim, Bakersfield, Los_Angeles, Riverside, SLO, and San_Diego. The input images were resized to `224 x 384`, which preserves the panoramic shape of the Street View images better than forcing them into a square image.

In terms of hidden layers, the network contains the standard MobileNetV3-Small feature extractor followed by a two-layer classifier head. The classifier head has one hidden linear layer with `1024` units and one output layer with `6` units. After the convolutional backbone, the model applies adaptive average pooling, then a fully connected layer from `576` features to `1024` hidden units, followed by the final linear layer from `1024` to `6` logits.

The model has `1,524,006` total parameters. The activations used in the network are primarily `ReLU` and `Hardswish` inside the MobileNetV3-Small backbone, along with `Hardsigmoid` inside the squeeze-and-excitation blocks. In the classifier head, I used a `Hardswish` activation after the hidden linear layer. I also used `Dropout(p = 0.2)` before the final output layer as regularization.

The final prediction pipeline also uses multi-view test-time augmentation. For each image, the model makes predictions on six views: the resized image, a horizontally flipped version, a left crop, a flipped left crop, a right crop, and a flipped right crop. The logits from these six views are averaged before taking the final class prediction. This does not change the number of model parameters, but it improves accuracy by making the final prediction less sensitive to the exact framing of the panorama.

I chose MobileNetV3-Small because it is compact enough to stay far below the project’s 50 MB size limit while still being expressive enough to learn useful visual cues from Street View images. The saved weight file for the final model is about 5.9 MB, and the final submission zip is still comfortably below the 50 MB limit.

**Training Procedure**

I trained the model in PyTorch using the `AdamW` optimization algorithm. The loss function was cross-entropy loss with label smoothing set to `0.05`. I trained on the provided labeled image set and created a stratified validation split containing 20% of the data so that each city remained roughly equally represented in both the training and validation sets.

Before feeding images into the model, I resized them to `224 x 384` and normalized them using the standard ImageNet channel means and standard deviations. During training, I applied data augmentation using random resized crops, random horizontal flips, and mild color jitter. These augmentations were intended to reduce overfitting and help the model become less sensitive to small visual changes across locations.

For the run used to create the final weights, I trained for `3` epochs with batch size `128`. I used a learning rate of `3e-4`, weight decay of `1e-4`, and a cosine annealing learning-rate schedule. During this run, I froze the pretrained backbone and trained the classifier head. The full training run took about `165.03` seconds total, which is approximately `2.75` minutes. The epoch times were about `62.40`, `50.95`, and `51.45` seconds respectively.

The training curve showed a steady decrease in empirical risk across epochs. The training loss decreased from `1.4451` in epoch 1 to `0.9435` in epoch 3, while the validation loss decreased from `1.2694` to `0.9368`. Validation accuracy also improved steadily from `54.60%` to `69.95%` to `72.84%`. This suggests that the model learned meaningful geographic features quickly and that transfer learning was important for getting strong performance with a relatively small network and a short training time.

After training, I evaluated the saved checkpoint with stronger inference-time averaging. A simple horizontal-flip average improved the validation score slightly, and the final six-view multi-view inference setup improved validation accuracy to `73.33%`. I also experimented with a stronger EfficientNet-B0 backbone, but it performed worse on the validation split than MobileNetV3-Small, so I kept the MobileNet model as the final submission.

Insert the training curve image from `training_curve.png` in this section.
