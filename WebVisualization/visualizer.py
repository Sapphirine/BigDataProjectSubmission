# Big Data Analytics
# Fall 2015
#
# Jie Yuan
# Ziyu He
# Yubin Shen

from flask import Flask, render_template, flash, redirect, request, session, abort, jsonify, json
import threading
app = Flask(__name__)

@app.route("/")
def index():
    return "Flask App!"

@app.route("/stuff", methods= ["GET"])
def stuff():
    with open("data/test.txt") as file:
        data = file.read()
    count = -1
    return jsonify(count=data)

@app.route("/test")
def test():
    return render_template('test.html')

# Generate the performance chart
@app.route("/d3vis/")
def d3test2():
    data = []
    with open("../NeuralNetwork/nn_mean_errors.txt") as file:
        for line in file:
            data.append(float(line.strip()))
    return render_template('linechart.html', data=json.dumps(data))

# Update the performance chart in real time
@app.route("/d3vis/stuff",methods= ["GET"])
def d3test2get():
    data = []
    with open("../NeuralNetwork/nn_mean_errors.txt") as file:
        for line in file:
            data.append(float(line.strip()))
    return json.dumps(data)
 
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)