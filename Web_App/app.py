from flask import Flask,request,render_template
import pandas as pd
import pickle

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def main():
    if request.method=='POST':
        file=open('../Models/RandomForest.pkl','rb')
        RF=pickle.load(file)
        file.close()

        N=request.form.get('n')
        P=request.form.get('p')
        K=request.form.get('k')
        temperature=request.form.get('temp')
        humidity=request.form.get('hum')
        pH=request.form.get('ph')
        rainfall=request.form.get('rain')

        df=pd.DataFrame([[N,P,K,temperature,humidity,pH,rainfall]],columns=['N','P','K','temperature','humidity','ph','rainfall'])
        prediction=RF.predict(df)[0]
    else:
        prediction=''
    return render_template('index.html',output=prediction)

if __name__=='__main__':
    app.run(debug=True)