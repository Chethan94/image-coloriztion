from email.mime import application
from flask import Flask, flash, request,  render_template,  jsonify
import tensorflow as tf
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.transform import resize
from PIL import Image
import io
import base64





application = Flask(__name__, static_url_path='')
application.secret_key = "123"
application.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpeg', 'jpg', 'webp'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

modelg=tf.keras.models.load_model('static/model/unetgan-dsgood1.h5')
def preandcolorimage(img):
    test_imgs=[]
    width, height = img.size
    img = img_to_array(img)
    img = resize(img ,(256,256))
    test_imgs.append(img)
    test_imgs = np.array(test_imgs, dtype=float)
    test_img = rgb2lab(1.0/255*test_imgs[:,:,:,0:3])[:,:,:,0]
    test_img = test_img.reshape(test_img.shape+(1,))
    output = modelg(test_img,training=False)
    output = output * 128
    for i in range(len(output)):
        result = np.zeros((256,256, 3))
        result[:,:,0] = test_img[i][:,:,0]
        result[:,:,1:] = output[i]
    output_img = lab2rgb(result)
    output_img_r = resize(output_img,( height,width))
    return output_img_r

def get_encoded_img(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return my_encoded_img

@application.route('/')
def upload_form():
    return render_template("index.html")

    
@application.route('/process', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file exist')
            return {
        'p': 'No file exist',
    }

        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return {
        'p': 'No file selected for uploading',
    }


        if file and allowed_file(file.filename):
            flash('File successfully uploaded')
            filestr = request.files['file']
            img = Image.open(filestr)
            output_img=preandcolorimage(img)
            output_imgd = Image.fromarray((output_img * 255).astype(np.uint8))
            oimg = get_encoded_img(output_imgd)
            response_data = {"p": 'success', "image": oimg,'filename': file.filename.rsplit('.', 1)[0].lower()+'_colorized.jpg'}
            return jsonify(response_data )
        else:
            flash('Upload only JPEG,JPG or PNG images only')
            return {
        'p': 'Upload only JPEG,JPG or PNG images only',
    }

@application.route('/<path:path>')
def static_file(path):
    return application.send_static_file(path)

if __name__ == "__main__":
    application.run()