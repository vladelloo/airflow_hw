import csv
import json
import os
from datetime import datetime
from glob import glob

import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')


def predict():
    def read_pkl_to_model():
        model_reference = os.listdir(f'{path}/data/models')[-1]
        with open(f'{path}/data/models/{model_reference}', 'rb') as file:
            model_pkl = dill.load(file)
        return model_pkl

    def open_test_jsons(json_path):
        with open(json_path, 'rb') as datafile:
            test_data = pd.json_normalize(json.load(datafile))
        return test_data

    model = read_pkl_to_model()
    predictions = pd.DataFrame()
    for datapath in glob(f'{path}/data/test/*.json'):
        df = open_test_jsons(datapath)
        prediction = pd.Series({'id': df.iloc[0]['id'], 'predict': model.predict(df)})
        predictions = pd.concat([predictions, prediction.to_frame().T], ignore_index=True)

    predict_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    predictions.to_csv(predict_filename, index=False)


if __name__ == '__main__':
    predict()
