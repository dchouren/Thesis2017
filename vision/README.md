# Vision

## File Organization
~/vision (or wherever you clone this) is the git repo. /spindle is where images/scratch/saved models should go. 

General file directory structure is root_dir/$TASK/$DATASET/

Task refers to the vision project. We built a menu classifier (menu), looked at thumbnail visual attractiveness (thumbnail), and looked at various restaurant categories (meals, cuisines, dishes).

Dataset refers to the dataset you are working with. So if you are training a model on scraped flickr images for thumbnail attractiveness, these would be written to $IMAGES/thumbnail/flickr.

I tried to be consistent with this task/dataset file organization throughout the repo. Output files were written to output/thumbnail/flickr/Uniquename, html files generated from this output are written to html/thumbnail/flickr/Uniquename, etc.

### vision/
The main directory is vision/. This contains src/, output/, resources/, html/, trained_models/, lua/. 

#### src/
See larger subsection

#### lua/
Harvard Lua model (not that good). Original code is here: https://jira.tripadvisor.com/browse/ML-474. Has its own README.

#### output/
Output from your models

#### ditto_results/
all*.json holds all the information retrieved from calls to Ditto's API. good*.json holds information about images classified by Ditto has good.

#### html/
I wrote examine_keras_output.py to format output csv files into html pages. This script outputs html files to html/.

#### debug/
Sometimes I wrote debug information from examine_keras_output.py to debug/.

#### trained_models/
Deprecated by $RESOURCES/${TASK}/${DATASET}/trained_models. I moved examined models to /spindle/resources/trained_models.


### /spindle/
I wrote images to  $IMAGES=/spindle/$USER/images
Resources should be written to $RESOURCES=/spindle/$USER/resources

#### /spindle/$USER/images/

Keras will generate images from a directory by searching for train and val directories. The subdirectories it finds in train/ and val/ will be the classes. These align to Keras' outputted labels according to alphabetic order.

$IMAGES/thumbnail/flickr/train/{flickr_food | food | nonfood} means we have three classes for this task. I did a lot of swapping of different datasets and dataset augmentation so my base $IMAGES/task/dataset directories are clogged with many datasets that I would swap in and out of train/val. 

I moved the final image sets I used to /spindle/images.


#### /spindle/$USER/resources/
/task/dataset/trained_models contains .h5 files which are the saved weights for a particular task. I stored both the weights of my top models and the weights of finetuned models here. Top models are small custom nets that you append to the end of a pretrained model. See build_top_model in bottleneck_train.py. For finetuned models, I took the base_model+top_model combination, froze a certain amount of starting layers, and then retrained the end layers. This takes much longer but in theory gives better results.

Base model weights can be found in $RESOURCES/weights


#### /spindle/resources/trained_models
Contains trained models for menu, human detection, and thumbnail quality.

#### /spindle/resources/weights
Contains .h5 weight files for base models

#### /spindle/resources/yml
Contains a .yml of my conda env

#### /spindle/images/
Contains menu/menu_receipt which has 9000 manually labeled 'menu', 'receipt', and 'other' images.
Contains thumbnail/{dslr,flickr} which has the final datasets I used for these models. There's also a lot of other stuff I tried that's left in my /spindle/dchouren/images/thumbnail/flickr directory. 


## Code

All the code is in src/. The main workflow is 
1) Execute query as found in queries/pull_geo.sh which writes paths for a given geo/placetypeid to resources/paths/task/dataset
2) format_image_paths.py, which formats paths for downloading and writes these to resources/fpaths/task/dataset
3) pull_train_val.sh, which downloads images to $IMAGES/task/dataset/{train,val}. Can expand dataset with augment_images.py.
4) bottleneck_train.py, which trains a top model on output from a pretrained base model
5) finetune.py, which finetunes layers from a base model + top model combo. Only works for VGG16 right now.
6) pull_testsets.sh, which downloads testsets
7) predict_X.py, which runs test images through a model trained by bottleneck_train.py or finetune.py and outputs results to output/
8) examine_keras_output.py, which formats output into an html file

This workflow is shown in run_menu.sh and run_thumbnail.sh.

You can also use scrape_flickr.py to scrape urls from Flickr, in which case you'd use format_flickr_paths.py instead of format_image_paths.py. 


### src/ditto
Contains some code used to format results with Ditto's third party classifier

#### fetch_ditto_results.py
Used to make calls to Ditto's API. Results are saved in ~/vision/ditto_results for Milan, Paris, and New York

### src/html
Contains a utils file and some files I used to generate html pages for different result files.

#### compare_results.py
Ad-hoc script to load Ditto results and two internal classifier results for the Ditto test

#### examine_keras_output.py
Used for the main output/ files generated by test_model.py. EXTREMELY ad-hoc. I changed the parsing based on what I needed.

#### html_utils.py
Useful methods for generating html code for a document.


### src/tools

#### convert_to_hdf5.py
Convert images to an hdf5 file.

#### download_images.sh
DEPRECATED BY 
Downloads images directly into train and val directories with a 90-10 split. Caps download at 32k images. I also just used cat file | parallel sudo curl -s -o #1-#2-#3-#4-#5. The formatted path files created by format_image_paths allow for curl's '#' naming convention. 

#### format_flickr/image/thumbnail_paths.py
Formats image paths from paths fetched from t_media so they can be named with curl's '#' naming convention. This is essential for ensuring unique filenames. If you don't have unique image names there might be problems later with improper classification. Flickr formatting only requires one '#', hence the separate file (passing a parameter was messier). Paths were written to ~/vision/resources/paths/$TASK/$DATASET. If you run into trouble with downloading images, check that you are splitting your paths file with the correct delimiter. 

#### label_data.py
Very useful for streaming images from a directory and allowing you to label it with key bindings. Labeling will move the image from the original directory to a directory. Key bindings/directories are hardcoded.

#### pull_image_paths.sh
Sample command for pulling image paths.

#### remove_unopenable_images.py
Remove images which can't be opened by Keras' preprocessing tools.

#### scrape_flickr.py
Scrape photos from Flickr API for a given search.

#### split_hdf5.py
Split hdf5 files that are too big (Lua has a 2G in-memory limit). 


### src/vision

#### run_thumbnail.sh
Rough script to pull all necessary images for training an internal model and testing on a list of geos. The main bottleneck_train and prediction parts have been tested but the image downloading might not work ot of the box.

#### run_menu.sh
Rough script to train an internal model for menu prediction. The main bottleneck_train and prediction parts have been tested but the image downloading might not work ot of the box. 

#### src/vision/models
Contains some stuff for loading stock models. Weights can also be found in $RESOURCES/weights.

#### augment_images.py
Uses Keras' ImageDataGenerator to create an augmented dataset by applying shears, rotations, flips, etc to images

#### bottleneck_train.py
One of the two heavy lifting model training scripts. Takes a pretrained net (weights for these can be found in $RESOURCES/weights) and runs your train/val images through. Saves the second to last layer of the net for each image into a .npy file. These are now considered to be a feature vector (bottlenecks) which are used to train a custom top model (a simple net). Intuitively what we are doing is extracting the features from each image judged by a pretrained net to be important, and using these to train a simple net. 

#### extract_bottlenecks.py
Save the feature vectors of images as produced by the second to last layer of a model. This is useful for doing batch predictions which is much faster than loading images and predicting one by one.

#### finetune.py
Takes a model and finetunes the last block (hardcoded). Much slower than bottleneck_train but will be more accurate in theory. Needs a small momentum for finetuning.

#### vision_utils.py
Keras utils that I packaged into one file. Has a _decode_predictions method that relies on a mapping between model predictions (0 indexed) to actual labels in alphabetic order. 

#### images_to_numpy.py
Script to save preprocess images from a directory and save them as a numpy array.

#### keras_utils.py
Utils written by Aaron Gonzalez on VR datascience, not really used but possibly useful.

#### mlutils.py
Utils written by someone on ML, not really used but possibly useful.

#### predict_one.py
This was my playground for loading a model and running quick tests on images in test_images.

#### quality_utils.py
Contains methods to get a blurriness metric (based on a Laplacian convolution), a cleanness metric, and face detction. 

#### predict_thumbnail.py
Loads a trained model and predicts thumbnail quality for images in a given fpath file one by one. Relies on all_maps in vision_utils to decode predictions. 

#### predict_menu.py
Menu prediction. Relies on all_maps in vision_utils to decode predictions. 

#### train_top_model.py
Also found in bottleneck_train.py. Factored out the actual top model training into separate file. 


### Future Work
- Right now we use a mix of finetuned models, which require loading a full model since the last layers have unique weights, custom top models, and classical vision techniques that operate on the original image such as blurriness and entropy metrics. Streamlining all the processing we want to do to operate on bottleneck feature representations will greatly speed up iteration time. The classical vision techniques require us to leave images in their original pixel format, and then if we want to work with bottleneck features, we have to run this image array through a base model. It will be easier and faster to use extract_bottlenecks.py to convert a testset into a vgg16 or resnet50 bottleneck feature .npy array. We can then use Keras' batch predict function, which is magnitudes faster than loading and preprocessing each image individually. But working with bottleneck representations means we can't use finetuned models either, since the bottleneck created by a finetuned model is unique to that model. 
- The human classifier model is based off of resnet50 bottleneck representations but all the work I did used vgg16. Either we can move to a consistent representation or simply save and load both vgg16 and resnet50 bottlenecks.
- If we have just one representation, the pipeline for prediction will look like this: [pull test images] -> convert test images to bottleneck.npy array using vgg16 or resnet50 (or something else) -> load .npy array -> batch_predict using as many top models as we want -> combine output from top models
- It should be relatively easy to use the classical vision techniques to train top models for entropy and blurriness
- For displaying html files and organizing results, I pulled locationid, thumbnailid information, loaded this into a python dict and then matched it with the locations I wanted to show thumbnails/other information for. It might be better to refine the initial Hive/psql queries to pull this all this information at the beginning rather than splicing it in later when you want to display an html page. See my use of load_thumbnails in examine_keras_output.py

### IMPORTANT NOTES
- When expanding to use places or base classifiers other than ones trained on imagenet, you'll have to write a new vision_utils.preprocess_input method since the hard-coded values for mean centering are tuned for imagenet.
- My path/fpath/output files followed the ordering [url,mediaid,locationid,...other,stuff,...prediction,probability] very strictly so I knew I could always reference these in that order.
- Mappings between Keras' predictions (0-indexed) and the actual class labels are stored in vision_utils.py's all_maps variable. I manually added mappings. I thought about automatically writing the mappings to a JSON file in a shared resources directory when you run bottleneck_train.py or finetune.py, but I decided I didn't want people accidentally overriding each other's mappings, hence forcing us to manually add to this dict. 
- Mean centering in vision_utils is using IMAGENET. If you transition to using base models trained on datasets other than Imagenet, you'll need to hardcode new mean-centering adjustments.



### Environment
I used Python 3.5.2 with the Anaconda environment 'daneel'. Download Anaconda with Python 3 and call 'source activate daneel'. I installed opencv3 separately and installed a fork of Keras (https://github.com/MarcBS/keras) from git with python setup.py develop so I could tweak some of the code (mostly for debugging purposes but also so I could print training progress--highly suggest doing this as some jobs might several hours). I also downloaded GNU parallel for parallel curl downloading of images and PyQt as a display backend for something. 
A .yml of my Conda env is in /spindle/resources/yml.


### Useful Links
- https://github.com/metalbubble/places365
- https://github.com/MarcBS/keras   # Keras fork with Caffe model conversion
- Walkthrough of bottleneck and finetuning: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
