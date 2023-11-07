import inquirer
import os
import glob
import pandas as pd
import open3d as o3d
import numpy as np
dbpath = r"./database/"


################ User prompt ###########################
questions = [
  inquirer.List('classes',
                message="from what class do you want to select an item?",
                choices=['AircraftBuoyant', 'Apartment', 'AquaticAnimal', 'Bed', 'Bicycle', 'Biplane', 'Bird', 'Bookset', 'Bottle', 'BuildingNonResidential', 'Bus', 'Car', 'Cellphone', 'Chess', 'City', 'ClassicPiano', 'Computer', 'ComputerKeyboard', 'Cup', 'DeskLamp', 'DeskPhone', 'Door', 'Drum', 'Fish', 'FloorLamp', 'Glasses', 'Guitar', 'Gun', 'Hand', 'Hat', 'Helicopter', 'House', 'HumanHead', 'Humanoid', 'Insect', 'Jet', 'Knife', 'MilitaryVehicle', 'Monitor', 'Monoplane', 'Motorcycle', 'Mug', 'MultiSeat', 'Musical_Instrument', 'NonWheelChair', 'PianoBoard', 'PlantIndoors', 'PlantWildNonTree', 'Quadruped', 'RectangleTable', 'Rocket', 'RoundTable', 'Shelf', 'Ship', 'Sign', 'Skyscraper', 'Spoon', 'Starship', 'SubmachineGun', 'Sword', 'Tool', 'Train', 'Tree', 'Truck', 'TruckNonContainer', 'Vase', 'Violin', 'Wheel', 'WheelChair'],
            ),
]
answers = inquirer.prompt(questions)
#print(answers["classes"])

#create list of items in chosen class
class_folder_path = os.path.join(dbpath, answers["classes"])
object_list = []
for obj_file_path in glob.glob(os.path.join(class_folder_path, '*.obj')):
    object_list.append(obj_file_path)
#print(object_list)

#ask for item to compare
questions = [
  inquirer.List('objects',
                message="Select an object in this class to retrieve the most similar items",
                choices=object_list,
            ),
]
input_object = inquirer.prompt(questions)
print(input_object["objects"])

questions = [
  inquirer.List('k',
                message="How many items do you want to retrieve?",
                choices=[2,3,4,5,6,7,8,9,10,15,20],
            ),
]
k = inquirer.prompt(questions)

print("the", k["k"], "most similar object to", input_object["objects"], "are: (first object is the query object itself)")

################# visualization functions ####################
def shift_mesh(mesh, distance):
    "shift all values of mesh to the right to display meshes next to eachother"
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) + np.array([0, distance, 0]))
    return mesh

def draw_meshes_results(result_list, draw_unit_cube=False, draw_coordinate_frame=False):
    """Draws a list of given meshes."""
    value = 0
    mesh_list = []
    for i in result_list:
        mesh1 = o3d.io.read_triangle_mesh(i+".obj")
        mesh1 = shift_mesh(mesh1, value)
        mesh_list.append(mesh1)
        value += 1.2
    o3d.visualization.draw_geometries(mesh_list, mesh_show_wireframe = True)

def vis(input_object, k):
    "match input onject using distance function -> make list of top k similar items -> display top k similar items"
    result_list = []
    #print("input object path:", input_object[1])
    result = match(input_object, k)
    for path in result:
        #print(path.split("/")[2].split("\\")[0])
        dbpath = r"./resampledO3D/"+ path.split("/")[2]
        result_list.append(dbpath)
    draw_meshes_results(result_list)
    print("This query returned ",same_class(result), "items belonging to the same class")


################## matching function (can be replaced by other matching function that returns dict of top k similar items) #############

#computes euclidean distance between 2 vectors
def euclidean_distance(vector1, vector2):
    euclidean = np.linalg.norm(np.asarray(vector1[2:9]) - np.asarray(vector2[2:9]))
    return euclidean

#computes cosine distance between 2 vectors
def cosine_distance(vector1, vector2):
    one_arr = np.asarray(vector1)
    two_arr = np.asarray(vector2)
    cosine = 1 - ((one_arr @ two_arr) / (np.linalg.norm(one_arr) * np.linalg.norm(two_arr)))
    return cosine

def scale_dict(my_dict, max_val):
    for i in my_dict:
        my_dict[i] = float(my_dict[i]/max_val)
    return my_dict 

#returns top k matches
def match(vector1, k):
    distance_dictionary = {}
    distance_dictionary_2 = {}
    for i in range(len(data)):
        vector2 = data.iloc[i]
        match_path = vector2[1]
        distance = cosine_distance(vector1[2:9], vector2[2:9])
        distance_dictionary[match_path] = distance


    # Sort the dictionary by value using a lambda function to extract the values
    result = dict(sorted(distance_dictionary.items(), key = lambda x: x[1], reverse = False)[:k])
    return result

def same_class(result):
    allkeys = list(result.keys())
    target = allkeys[0]
    target = target.split("/")[2].split("\\")[0]    
    count = 0
    for path in allkeys[1:]:
        i = path.split("/")[2].split("\\")[0]
        if i == target:
            count += 1
    return count

###############################################################

data = pd.read_pickle("normalized_features_final.pkl")
# rewrite path name in order to match to the pkl feature database
path = "./features/"+input_object["objects"].split("/")[2].split(".")[0]
value = data.loc[data["path"] == path]


#visualise the top k similar objects
k = k["k"] +1
vis(value.squeeze(), k)





