import json
import an_source
from flask import Flask, render_template, url_for, request
from datetime import datetime
app = Flask(__name__) 

@app.add_template_filter
def today(fecha):
    return fecha.strftime('%d/%m/%Y')

@app.route('/', methods = ['GET','POST'])
def index():
    if request.method == 'POST':
        x = request.form['x1']
        return f"Valor de X: {x}"
    date = datetime.now()
    data = an_source.cant_v_unic.to_dict()
    labels = list(data.keys())
    datos = list(data.values())
    print("Desde app",data)
    return render_template('index.html',
                           labels = labels,
                            datos = datos,
                           fecha = date)

if __name__ == '__main__':
    app.run(debug=True)
    
