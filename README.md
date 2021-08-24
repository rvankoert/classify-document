# classify-document
Tool for classification of documents based on their layout.

Steps:
gather some training data and label this by putting images in their respective folders. For example make two folders: letters and forms.
traindir -> letters
traindir -> forms

optionally split the training data into train and validation so you have the following folders
traindir -> letters
traindir -> forms
validationdir -> letters
validationdir -> forms
Make sure none of the images are duplicated between train and validation


Commands are Case Sensitive!
train a model:
python3.6 main.py --train True --train_set /home/rutger/data/republic/train/ \
 --validate True \
 --validation_set /home/rutger/data/republic/validation/ \
 --seed 42 \
 --gpu 0

Look at the numbers of the validation. They should be getting more correct each few epochs.


Inference a model:
make sure the data is in a directory called "data" like this
/home/rutger/data/republic/data
and then call like this:
python3.6 main.py --inference True --inference_set /home/rutger/data/republic/


# classify-document
