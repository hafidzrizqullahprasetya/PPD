"""
Created on 2020-11-25
@author: Dr. Ganjar Alfian
email : ganjar@dongguk.edu
for teaching purpose only.
"""

from typing import Optional

import pandas as pd
import joblib
from flask import Flask, redirect, url_for, request, render_template

app = Flask(__name__)
#load index.html/ first page. receive input variable from user
@app.route("/tabular/")
def index():
    return render_template('index.html')

#load result.html. the result of prediction is presented here.
@app.route('/tabular/result/', methods=["POST"])
def prediction_result():
    #receiving parameters sent by client
    age = request.form.get('age')
    sex = request.form.get('sex')
    alb = request.form.get('alb')
    alp = request.form.get('alp')
    alt = request.form.get('alt')
    ast = request.form.get('ast')
    bil = request.form.get('bil')
    che = request.form.get('che')
    chol = request.form.get('chol')
    crea = request.form.get('crea')
    ggt = request.form.get('ggt')
    prot = request.form.get('prot')
    #load the trained model.
    filename = 'dt_model.model'
    loaded_model= joblib.load(filename)
    # helper converters
    def _to_float(x: Optional[str]) -> float:
        return float(x) if x is not None and x != '' else float('nan')

    def _encode_sex(x: Optional[str]) -> int:
        if x is None:
            return 0
        s = str(x).strip().lower()
        if s in ('m', 'male', '1', 'true'):
            return 1
        if s in ('f', 'female', '0', 'false'):
            return 0
        try:
            return int(float(s))
        except Exception:
            return 0

    sex_enc = _encode_sex(sex)

    # create new dataframe matching training feature order
    # Training used: df_X_new = pd.concat([df_X['Sex'], df_X_numb], axis=1)
    columns_order = ['Sex', 'Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
    df_input = pd.DataFrame(columns=columns_order)
    df_input.loc[0] = [
        sex_enc,
        _to_float(age),
        _to_float(alb),
        _to_float(alp),
        _to_float(alt),
        _to_float(ast),
        _to_float(bil),
        _to_float(che),
        _to_float(chol),
        _to_float(crea),
        _to_float(ggt),
        _to_float(prot),
    ]
    #make prediction
    #print(df_input)
    result = loaded_model.predict(df_input)
    #print(result)
    for i in result:
        int_result = int(i)
        if(int_result==0):
            decision='Suspect Blood Donor'
        elif (int_result==1):
            decision='Hepatitis'
        elif (int_result==2):
            decision='Fibrosis'
        elif (int_result==3):
            decision='Cirrhosis'
        else:
            decision='Not defined'
    #print('Disease is ', decision)
    #return the output and load result.html
    return render_template('result.html', age=age, sex=sex, alb=alb, alp=alp, alt=alt, ast=ast, bil=bil,
                           che=che, chol=chol, crea=crea, ggt=ggt, prot=prot, status=decision)

if __name__ == "__main__":
    #host= ip address, port = port number
    #app.run(host='127.0.0.1', port='5001')
    app.run()
