## Welcome to Plant Desease Detection API

An API that predicts plant deseases based on leaf images from this [Kaggle challenge](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)

## Prerequisites

Download the dataset from [here](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset) and put it into the following
folder structure:

```
plant_detection
│
└───dataset
    │
    └───test
    │    │   AppleCedarRust1.JPG
    │    │   AppleCedarRust2.JPG
    │    │   ...
    │
    └───train
    │    │   
    │    └───Apple__Apple_scab
    │    └───Apple__Apple_rot
    │    └───...
    │
    └───valid
    │    │   
    │    └───Apple__Apple_scab
    │    └───Apple__Apple_rot
    │    └───...
   

```


Create a new virtual environment first by executing `make setup`. Secondly, install [Node.js](https://nodejs.org/en/) for
running the application. However, this is not required if you want to use the API only.

Making predictions for new images requires trained model files within `/models`. You can execute the training script
`python run.py` for training the models defined within `src/models.py`. By default the batch size is 100 and the amount of epochs equals 20. 
Feel free to adjust the parameters and models as you desire.

## Run API only

Within the project root directory run the API within you CLI as follows:

```shell
uvicorn api:api --reload
```

Access FastAPI UI by open your browser at http://127.0.0.1:8000/docs. Open the POST method tab and click on *Try it out*.
You can manipulate the JSON request body as you desire.

The default JSON request body:
```json
{
  "model_path": "./models/my_cnn'",
  "img_path": ",./dataset/test/PotatoHealthy1.JPG"
}
```

## Run Dashboard and API locally

To host the dashboard and API locally run the following command within project root directory:

```shell
make run_dev
```

The dashboard should start at http://localhost:5000/, while the API is running at http://localhost:8000/. To interact with the API only 
check out the previous section.




