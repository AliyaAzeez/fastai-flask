import boto3
import os

def download_dir(s3_key, local_folder, s3_bucket_name, aws_access_key_id, aws_secret_access_key):
    """
    Function to download the data from s3 bucket for training

    Args:
        s3_key(str): pattern to match in s3
        local_folder(str): local path to folder in which to place files
        s3_bucket_name(str): s3 bucket with target contents
        aws_access_key_id(str): AWS access key id stored in .env file
        aws_secret_access_key(str): AWS secret access key stored in .env file

    Returns:
        True 
    """
    s3_client = boto3.client('s3',
                             aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key )
    keys = []
    dirs = []
    next_token = ''
    base_kwargs = {
        'Bucket':s3_bucket_name,
        'Prefix':s3_key,
    }
    while next_token is not None:
        kwargs = base_kwargs.copy()
        if next_token != '':
            kwargs.update({'ContinuationToken': next_token})
        results = s3_client.list_objects_v2(**kwargs)
        contents = results.get('Contents')
        for i in contents:
            k = i.get('Key')
            if k[-1] != '/':
                keys.append(k)
            else:
                dirs.append(k)
        next_token = results.get('NextContinuationToken')
    for d in dirs:
        dest_pathname = os.path.join(local_folder, d)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
    for k in keys:
        dest_pathname = os.path.join(local_folder, k)
        if not os.path.exists(os.path.dirname(dest_pathname)):
            os.makedirs(os.path.dirname(dest_pathname))
        s3_client.download_file(s3_bucket_name, k, dest_pathname)

    return True
        
def upload_file_s3(s3_bucket_name,local_folder,s3_key,file_name,aws_access_key_id,aws_secret_access_key):
    """ 
    Function to upload a file in s3 bucket

    Args:
        s3_bucket_name(str): s3 bucket name to which the model and confusion matrix is to be uploaded
        local_folder(str): path of the file in the local system
        s3_key(str): path to the location in s3 bucket
        file_name(str): filename to be saved in s3

    Returns: 
        Object url(str): s3 object url of the uploaded file
    """
    print("uploading model to s3")
    s3 = boto3.resource('s3',
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key)
    s3.Bucket(s3_bucket_name).upload_file(local_folder,'%s/%s' %(s3_key,file_name))
    bucket_location = boto3.client('s3').get_bucket_location(Bucket=s3_bucket_name)
    key_name = f'{s3_key}/{file_name}'
    object_url = "https://s3-{0}.amazonaws.com/{1}/{2}".format(
                            bucket_location['LocationConstraint'],s3_bucket_name,key_name)
    return object_url