from PIL import Image
import numpy as np
import io
from keras.models import model_from_json
from fastapi import FastAPI, File, UploadFile
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title='Predictor API')

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Preparing image for the crowd classifier
def prepare_image(file):
    img = Image.open(io.BytesIO(file)).resize((224,224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    return x


#Classifier predictor function
def classifier_predict(file):
    img = prepare_image(file)
    vgg16_saved = load_model('models/model_new.h5')
    p = vgg16_saved.predict(img)
    arr = ["Crowded", "Heavily_Crowded", "Light_Crowded", "Normal", "Semi_Crowded"]
    l = p[0].copy()
    l.sort()
    m = max(p[0])
    k=""
    for i in range(len(p[0])):
        if m == p[0][i]:
            k = str(arr[i])  
    return k


#Loading crowd counting model
def load_model2():
    # Function to load and return neural network model 
    json_file = open('models/Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("weights/model_A_weights.h5")
    return loaded_model

#Function to load,normalize and return image for crowd counting model
def create_img(file): 
    im = Image.open(io.BytesIO(file)).convert("RGB")
    im = np.array(im)
    im = im/255.0
    im[:,:,0]=(im[:,:,0]-0.485)/0.229
    im[:,:,1]=(im[:,:,1]-0.456)/0.224
    im[:,:,2]=(im[:,:,2]-0.406)/0.225

    im = np.expand_dims(im,axis  = 0)
    return im

#Function to load image,predict count
def predict(file):
    model = load_model2()
    image = create_img(file)
    ans =   model.predict(image)
    count = np.sum(ans)
    count = round(float(count))
    return count


@app.get('/index')
async def hello_world():
    return "hello world"



@app.post("/predictClass")
async def classify_image(file: UploadFile = File(...)):
    val = classifier_predict(await file.read())
    return val

@app.post("/predictCount")
async def crowd_count(file: UploadFile = File(...)):
    val = predict(await file.read())
    return val