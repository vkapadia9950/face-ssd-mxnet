# Fire-SSD-MXNet
Fire Detector trained by MobileNet SSD


## 0. Installation
* Install MXNet framework and GluonCV toolkit
	* For CPU only:
	
		`pip install mxnet gluoncv`
	
	* For GPUs
		
		`pip install mxnet-cu90 gluoncv`
    	> Change to match with CUDA version. `mxnet-cu100` if CUDA 10.0 is installed

## 1. Prepare Dataset
### 1.1 Image dataset
* Training Fire Images are available in Training_Images folder and corresponding annotations are in Training_Annotation folder


* Validation Fire Images are available in Validation_Images folder and corresponding annotations are in Validation_Annotation folder


### 1.2 Create record data
Run `lst-rec-prep.py` to create List and RecordIO file - they are the dataset format developed by MXNet. To support the training procedure, it is preferred to utilize data in binary format rather than raw images as well as the annotations.

If you are not familiar with this process, refer this tutorial [[link]](https://gluon-cv.mxnet.io/build/examples_datasets/detection_custom.html)
#### 1.2.1 Create LST file
Follow step 0 to step 2 to first create `.lst` file. By performing these steps, 2 file `train_data.lst` and `val_data.lst` are generated.

#### 1.2.2 Create REC file
After obtaining the `.lst` files. Start generating `.rec` by using built-in feature from MXNet

`python im2rec.py train_data.lst Training_Images --pass-through --pack-label`

`python im2rec.py val_data.lst Validation_Images --pass-through --pack-label`

It's gonna take a few seconds to create record files for `train` and `val` datasets. After finishing this step, 4 files are created:
* train.idx
* train.rec
* val.idx
* val.rec

Move all generated files to `datasets` directory

`mkdir datasets`

`mv *.lst *.idx *.rec datasets`

## 2. Train SSD-Face
After the preparation process is successfully done. The training procedure can be started by the following command:

`python train_ssd.py`

This is the full training program originally from MXNet [[link]](https://gluon-cv.mxnet.io/build/examples_detection/train_ssd_voc.html).

In this program, fire detector is trained by fine-tuning technique and the object detection pretrained model is trained from VOC dataset.

`import gluoncv as gcv`

`TARGET_CLASSES = ['fire']`

`net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', 
                                  classes=TARGET_CLASSES,
                                  pretrained_base=False,
                                  transfer='voc')`

Checkpoints are saved every 10 epochs and the best model with the best accuracy in `val` set is also stored and updated after every single epoch.

## 3. Results:
After processing the training procedure, 3 files are obtained:
* ssd_512_mobilenet1.0_voc_best_map.log
* ssd_512_mobilenet1.0_voc_best.params
* ssd_512_mobilenet1.0_voc_train.log

`test-fire-detector.py`is the Python Script developed to test the performance of trained Fire detection model

Results of Fire Detector are Stored in results folder
