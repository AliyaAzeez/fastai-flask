# from flask import Flask
# from keras.models import Sequential, load_model
# app = Flask(__name__)
# lr_model = load_model('jvp004_lr_2.h5')
# import jvc_app.views

from os import environ 

AWS_ACCESS_KEY_ID = environ.get('DJANGO_AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = environ.get('DJANGO_AWS_SECRET_ACCESS_KEY')