from flask import Flask,render_template,request
import pickle
import numpy as np
app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1],2)
    if float(output) > 0.5:
        return render_template('index.html',pred='You are in danger.\nProbability of diabetes is {} %'.format(float(output)*100))
    else:
        return render_template('index.html',pred='Congrats! You are safe.\n Probability of diabetes is {} %'.format(float(output)*100))

if __name__ == '__main__':
    app.run(debug=True)



