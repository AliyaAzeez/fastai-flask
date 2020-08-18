from flask import Flask, escape, request,render_template
from fastai.basics import *
from PIL import Image
from torchvision import *
from fastai.vision import *
import torch
import PIL
from datetime import datetime
import pytz
import cv2
from .s3_ops import download_dir, upload_file_s3
from .utils import find_appropriate_lr, plot_save_confusion_matrix
from flask import jsonify,request

app = Flask(__name__)
app.config.from_pyfile('__init__.py')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/fastai-train',methods=['GET','POST'])
def train_model():
    """
    Endpoint to train fastai models and upload the models in AWS S3 bucket

    Datasets can be downloaded from AWS S3 to local folder for training and the option to upload the model to S3 bucket.
    AWS credentials stored in .env file and accessed using python-dotenv.
    
    Arguments in the post request:
        model_name(str): name of the created model
        data_path(str): local path to training data
        export_path(str): local path to save model
        valid_pct(float): validation percentage
        size(int): image size
        bs(int): batch_size
        to_normalize(bool): Boolean input to normalize or not
        num_epoch(int): number of epochs
        to_download_data(bool): Boolean input to download data from aws s3 bucket
        to_upload_model(bool): Boolean input whether to upload the trained model to s3 or not
        bucket_name(str): s3 bucket name
        model_prefix(str): path to the location in s3 bucket to save model
        cm_prefix(str): path to the location in s3 bucket to save confusion matrix
        local_data_dir(str): directory path to download the data from s3
        train_data_s3_key(str): s3 object key to data 

    Returns:
        if to_upload_model is TRUE, returns

            Model_S3_Url: Model S3 object url 
            CM_S3_Url: Confusion Matrix S3 object url

        returns Success otherwise.

    """

    if request.method == 'POST':
        data = request.get_json()
        model_name = data.get("model_name")
        data_path = data.get("data_path")
        export_path = data.get("export_path")
        valid_pct = data.get("valid_pct")
        size = data.get("size")
        bs = data.get("bs")
        to_normalize = data.get("to_normalize")
        num_epoch = data.get("num_epoch")
        to_download_data = data.get("to_download_data")
        to_upload_model = data.get("to_upload_model")
        bucket_name = data.get("bucket_name")
        model_prefix = data.get("model_prefix")
        cm_prefix = data.get("cm_prefix")
        local_data_dir = data.get("local_data_dir")
        train_data_s3_key = data.get("train_data_s3_key")
        base_arch = models.resnet50
        tfms = [[rotate(degrees=(-5,5))],[rotate(degrees=(-5,5))]] 
        
        # aws credentials saved in .env file
        aws_access_key_id = app.config.get("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = app.config.get("AWS_SECRET_ACCESS_KEY")
        
        if to_download_data:
            print(train_data_s3_key,local_data_dir,bucket_name)
            print("downloading data")
            download_dir(s3_key=train_data_s3_key, local_folder=local_data_dir, s3_bucket_name=bucket_name, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        data = ImageDataBunch.from_folder(path=data_path, valid_pct=valid_pct,ds_tfms=tfms, size=size, bs=bs)
        
        if to_normalize:
            data=data.normalize()
        learn = cnn_learner(data, base_arch, metrics=accuracy, pretrained=True)
        learn.fit(num_epoch)
        lr=find_appropriate_lr(learn)
        
        #find_appropriate_lr plot is set to false 
        ## Option to take epoch as parameter
        
        learn.unfreeze()
        learn.fit_one_cycle(4,lr)
        
        
        dt_now = datetime.now(pytz.timezone("UTC"))
        now = dt_now.strftime("%Y_%m_%d__%H_%M")
        
        model_file_name=f'{model_name}_{now}.pkl' 
        model_export_path=f'{export_path}{model_file_name}'
        
        # save the model 
        learn.export(model_export_path)
        
        class_report=ClassificationInterpretation.from_learner(learn)
        cm_file_name=f'{model_name}_{now}_cm.png'
        cm_export_path=f'{export_path}{cm_file_name}'
        plot_save_confusion_matrix(class_report,cm_path=cm_export_path) # function that plot and save confusion matrix 

        #upload model and confusion matrix to s3: 
        if to_upload_model:
            upload_model_obj_url=upload_file_s3(bucket_name,model_export_path,model_prefix,model_file_name,aws_access_key_id,aws_secret_access_key)
            upload_confusion_obj_url=upload_file_s3(bucket_name,cm_export_path,cm_prefix,cm_file_name,aws_access_key_id,aws_secret_access_key)
            return {
                "model_s3_url":upload_model_obj_url,
                "cm_url":upload_confusion_obj_url
            }

        return "Success"
        
