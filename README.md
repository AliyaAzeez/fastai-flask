# Fastai-Flask

Flask app to train fastai models. 

The training dataset can be downloaded from the AWS S3 bucket.
Training starts when the request with required parameters is send. The example request is added in the jupyter notebook in this repo. The trained model along with confusion matrix is uploaded to the S3 bucket and returns the s3 object url.
