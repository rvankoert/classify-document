# classify-document
Tool for classification of documents based on their layout.

Steps:
gather some training data and label this by putting images in their respective folders. For example make two folders: letters and forms.
* `${traindir}/letters`
* `${traindir}/forms`


optionally split the training data into train and validation so you have the following folders:
* `${traindir}/letters`
* `${traindir}/forms`
* `${validationdir}/letters`
* `${validationdir}/forms`

Make sure none of the images are duplicated between train and validation


## Train a model:
```
python3.6 main.py --do_train \ 
 --train_set ${traindir} \
 --do_validation \
 --validation_set ${validationdir} \
 --seed 42 \
 --gpu 0
```
Look at the numbers of the validation. They should be getting more correct each few epochs.


## Inference a model:

The call will look like this:
```
python3 main.py --do_inference --inference_set /path/to/inference/dir/ --existing_model /path/to/model/best_val
```

if you have data that is not balanced (different numbers of items per class) it might make sense to add 
`--use_class_weights`
