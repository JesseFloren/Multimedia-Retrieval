from flask import Flask, request, jsonify
import open3d as o3d
import resampling as res
import feature_extraction as fe
import custom_distance_function as q
import numpy as np


app = Flask(__name__)

@app.route("/", methods=["POST"])
def query_object():
    content = request.json
    mesh = o3d.io.read_triangle_mesh(content['path'])
    pp_mesh = res.resample_single_file(mesh)
    feature_vector = fe.get_feature_vector(pp_mesh)
    results = q.query_feature_file(feature_vector)
    return jsonify({"results": results})

@app.route("/feature", methods=["POST"])
def query_object_features():
    content = request.json
    feature_vector = np.load(content['path'], allow_pickle=True)
    results = q.query_feature_file(feature_vector)
    return jsonify({"results": results})

# print(query_object("./database/Bicycle/D00016.obj"))

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True)