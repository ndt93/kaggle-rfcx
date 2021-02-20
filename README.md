# kaggle-rfcx
Notebooks and codes for the Kaggle RFCX competition: https://www.kaggle.com/c/rfcx-species-audio-detection

# Preprocessing
* Crop the full length audio clip into smaller clips
* Convert each small clip into series of mel-spectrograms
* Apply computer vision backbone with various time/frequency aware top layers on the mel-spectrograms

# Model Architectures
* EfficientNetB0 with custom top 1D convolution and self-attention layers for both framewise and clipwise audio events detection
* ResNet34 Backbone with top global max pooling and FN layers
* ResNet50 Backbone with top global max pooling and FN layers
* PANN architecture with customised self mix-up augmentation

# Ensemble
* 20 models, 5 folds each from the above 4 architectures
* Combine results using geometric means of predictions

# Augmentations used
* Gaussian noise and Gaussian SNR
* Random volume gain
* Horizontal and vertical flips (only for Resnet34 and Resnet50 models)
* Mix-up
* Random shift
* Spec-augmentation

# Loss functions
* LSEP loss for Resnet34 and Resnet50
* Soft Macro F1 loss for EfficientNet + self attention model
* Custom Binary Cross-entropy loss with auxilary loss on framewise predictions for PANN model

# Validation metrics
* Label-weighted label-ranking average precision
* ROC AUC

# Training Techniques
* Schedule learning rate using cosine Annealing with warmpups and restarts
* Adam optimizer
* Freeze backbone -> train top layers for 5 epochs with high learning rate -> unfreeze all layers except batch normalization -> train with early stopping
