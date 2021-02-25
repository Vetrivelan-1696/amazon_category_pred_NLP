from flask import Flask,request,jsonify,render_template
app = Flask('Category Model')
app.config["TEMPLATES_AUTO_RELOAD"] = True

import tensorflow as tf  
import pickle
import numpy as np
import requests


from tensorflow.keras.preprocessing.sequence import pad_sequences


global category_labels
global labels

category_labels={'petsupplies': 0, 'beauty': 1, 'games': 2, 'food': 3, 'office': 4}
labels=list(category_labels.keys())




max_length=100
trunc_type='post'
padding_type='post'



new_model = tf.keras.models.load_model(r'C:\Users\Admin\Desktop\python\Deeplearning\NLP_FLASK_APP\models\bilstm_model.h5')
with open(r'C:\Users\Admin\Desktop\python\Deeplearning\NLP_FLASK_APP\models\tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)




def predict_category():
        txt={"my office laptop has overloaded"}
        # seq=loaded_tokenizer.texts_to_sequences({text})
        # print(text)
        
        txt={"Office chair is not comfortable"}
        seq=loaded_tokenizer.texts_to_sequences(txt)
        padded=pad_sequences(seq,maxlen=max_length,padding=padding_type,truncating=trunc_type)
        check2=new_model.predict(padded)
        print(check2)
        
        print("labels:",labels[np.argmax(check2)])
        return labels[np.argmax(check2)]

@app.route('/')
def home():
    return predict_category()
    
        
        



    


# @app.route('/')
# def home():
#     return render_template('form.html')  # user to enter an input 

# @app.route('/result', methods=['POST'])  # take input and predict output and show the user
# def result():
#     if request.method == 'POST':
#       text = request.form['input']
#       predicted_category= predict_category(text)
#       return render_template('result.html', text=text,   predicted_category=predicted_category)



if __name__ == '__main__':
   app.run(host="localhost",port=9994,debug=True)