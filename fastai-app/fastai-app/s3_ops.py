import boto3
import os

def download_dir(s3_key, local_folder, s3_bucket_name, aws_access_key_id, aws_secret_access_key):
    """
    params:
    - s3_key: pattern to match in s3
    - local_folder: local path to folder in which to place files
    - s3_bucket_name: s3 bucket with target contents
    - client: initialized s3 client object
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
        
def upload_file_s3(s3_bucket_name,local_folder,s3_key,file_name,aws_access_key_id,aws_secret_access_key):
    """ 
    Function to upload a file in s3 bucket
    Input:s3_bucket_name: s3 bucket name
          local_folder: path of the file in the local system
          s3_key: path to the location in s3 bucket
          file_name: filename to be saved in s3
    Returns: Object url: s3 url of the file
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