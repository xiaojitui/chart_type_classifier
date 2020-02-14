# chart_type_classifier

This is a CNN model used to classify the type of a chart (e.g. LineGraph, BarGraph, BoxPlot, ...)

Download training datasets (https://github.com/arpitjainds/Chart-Image-Classification) and save all images into 'datasets' folder

Use "train.py" to train the classifier. Two CNN models (Resnet & Inception) are available as baselines for fine tuning. 

Download pre-trained Resnet & Inception models (https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) and save weights into 'saved_weights' folder

After the training, the processed data will be saved in 'processed_datasets' folder. The fine tuned models will be saved in 'fine_tuned' folder. 

