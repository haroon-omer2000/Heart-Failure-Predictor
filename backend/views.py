
from flask import Flask, jsonify, Blueprint, request
import controller

app = Flask(__name__)


@app.route("/make_prediction", methods=['POST'])
def make_prediction():

    details = request.get_json()

    controller_obj = controller.Controller_Class()
    results = controller_obj.evaluate(details)

    return jsonify(results)


app.run(debug=True)
