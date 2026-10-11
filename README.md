German Traffic Sign Recognition with CNN
An end-to-end computer-vision project that classifies German traffic-sign images into 43 classes using a convolutional neural network (CNN). The notebook walks through data inspection, image preprocessing, augmentation, model training, validation diagnostics, and evaluation on the official GTSRB test set.
Notebook: GTSRB_CNN.ipynb
Framework: TensorFlow / Keras
Task: Multi-class image classification

Project overview
Traffic-sign recognition is a computer-vision task in which a model learns to identify a sign from its visual appearance. This project builds a CNN from scratch and evaluates it with both overall metrics and class-level error analysis, rather than relying on accuracy alone.
Pipeline
Dataset and annotations
        ↓
Data inspection and class counts
        ↓
Train / validation split
        ↓
RGB conversion → ROI crop → resize to 30 × 30 → normalization
        ↓
Training-only image augmentation
        ↓
CNN training with early stopping
        ↓
Validation metrics and error analysis
        ↓
Official test-set evaluation
Dataset
The project uses the German Traffic Sign Recognition Benchmark (GTSRB). The benchmark contains 43 traffic-sign classes, more than 50,000 images overall, and training images organized into tracks; each track contains multiple images of one physical traffic sign.
The notebook expects a dataset directory containing:
- Train.csv — image paths, class IDs, and region-of-interest (ROI) coordinates for training images.
- Test.csv — image paths, class IDs, and ROI coordinates for the official test images.
- The image files referenced by the CSV Path column.
Dataset reference: German Traffic Sign Recognition Benchmark (GTSRB).
The dataset is not included in this repository. Attach or download it separately and configure the path in the notebook.
Methodology
1. Data understanding
The notebook inspects dataset metadata, sample counts, and per-class representation. It also visualizes preprocessed examples to check that image reading, channel conversion, cropping, and labels look reasonable.
2. Validation split and leakage awareness
The notebook uses scikit-learn's StratifiedGroupKFold to form one train/validation holdout from the training metadata. The intention is to keep related frames from a physical-sign track on the same side of the split while preserving class representation as far as the group constraint allows. The official test set is reserved for final evaluation.
Important pre-publication check: in the currently saved notebook, the active track_id expression uses os.path.basename(p).split('_')[1], while the expression using [0] is commented out. For standard GTSRB filenames of the form track_frame, the first component identifies the track and the second identifies the frame. To make the groups correspond to physical-sign tracks, use this expression in the split cell:
train_df['track_id'] = (
    train_df['ClassId'].astype(str) + '_' +
    train_df['Path'].apply(
        lambda p: os.path.basename(p).split('_')[0]
    )
)
Then rerun the notebook from the split cell onward and verify that the train/validation group intersection is empty. The currently displayed split counts and validation metrics were generated with the active expression in the uploaded notebook; regenerate them after correcting the group identifier before presenting them as results from a track-isolated evaluation.
3. Image preprocessing
Each image is processed consistently:
1. Read the image from disk with OpenCV.
2. Convert OpenCV's BGR channel order to RGB.
3. Crop to the annotated ROI using coordinates from the CSV file.
4. Resize the crop to 30 × 30 pixels.
5. Convert pixel values to float32 and scale them to the range [0, 1].
6. One-hot encode class labels for categorical cross-entropy training, while retaining integer labels for scikit-learn metrics.
4. Data augmentation
Augmentation is applied to training batches only:
- Rotation: up to 10 degrees
- Zoom: 0.15
- Width and height shifts: 0.1
- Shear: 0.1
- Horizontal flipping: disabled
The validation and official test images are not augmented, so their metrics are calculated on the original preprocessed samples.
5. CNN architecture and training
The notebook trains a CNN using TensorFlow/Keras. The model includes:
- Convolutional layers to learn edges, borders, textures, and more complex visual patterns.
- Max pooling to reduce spatial dimensions.
- Dropout to help reduce overfitting.
- A dense layer to combine learned features.
- A 43-unit softmax output for class probabilities.
Training configuration in the notebook:
Parameter	Setting
Input size	30 × 30 × 3
Number of classes	43
Batch size	128
Maximum epochs	40
Optimizer	Adam
Learning rate	0.00110
Loss	Categorical cross-entropy
Early stopping	Monitors validation loss; patience 5; restores best weights


The learning rate and architecture are the selected configuration recorded in the notebook; the notebook does not rerun the earlier hyperparameter search.
6. Evaluation and error analysis
The notebook reports:
- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Per-class precision, recall, F1-score, and support
- Confusion matrix
- Classes with the lowest validation F1-score
- Examples of misclassified validation images
Macro metrics are included because class counts differ: macro averaging gives each class equal weight instead of allowing high-frequency classes to dominate the aggregate score. Error examples help identify cases for further investigation, but individual errors alone do not establish their cause.
Results currently shown in the notebook
The saved notebook output reports the following metrics:
Split	Accuracy	Macro precision	Macro recall	Macro F1
Validation	98.85%	98.72%	98.46%	98.54%
Official test	98.55%	97.21%	97.95%	97.50%

Reference
J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel, “The German Traffic Sign Recognition Benchmark: a multi-class classification competition,” Proceedings of the IEEE International Joint Conference on Neural Networks (IJCNN), 2011, pp. 1453–1460.
Dataset website: https://benchmark.ini.rub.de/gtsrb_dataset.html
