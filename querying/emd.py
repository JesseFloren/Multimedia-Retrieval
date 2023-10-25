import open3d as o3d
import csv


def get_features(features_path, mesh_path):
    with open(features_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            filepath = row[0]
            if filepath == mesh_path:
                label = row.pop(1)
                row.pop(0)
                features = [float(feature.replace(" ", "")) for feature in row if feature != " "]
                return features, label
        else:
            raise RuntimeError(f"Mesh path {mesh_path} not found in mini-database.")


def load_meshes(meshpaths):
    # Load meshes
    meshes = []
    for i, meshpath in enumerate(meshpaths):
        mesh = o3d.io.read_triangle_mesh("miniDB/" + meshpath)
        mesh.compute_vertex_normals()

        # Add translation offset
        mesh.translate((i, 0, 0))
        meshes.append(mesh)

    return meshes


def visualize(meshes):
    o3d.visualization.draw_geometries(
        meshes,
        width=1280,
        height=720,
        mesh_show_wireframe=True
    )


def get_emd(features_1, features_2):
    i, j = 0, 1
    flow = [[0 for _ in range(len(features_1))] for _ in range(len(features_1))]
    difference = [0] * len(features_1)
    row = [0] * len(features_1)

    # Initialize empty flow matrix
    for p in range(len(features_1)):
        flow[p] = row.copy()
        flow[p][p] = min(features_1[p], features_2[p])
        difference[p] = features_1[p] - features_2[p]

    # Fill out the flow matrix by spreading differences
    while i + j < 2 * (len(features_1) - 1):
        if difference[i] > 0 and difference[j] < 0:
            if difference[i] <= -difference[j]:
                flow[j][i] = difference[i]
                difference[j] += difference[i]
                difference[i] = 0
                i += 1
                j = i + 1
            else:
                flow[j][i] = -difference[j]
                difference[i] += difference[j]
                difference[j] = 0
                if j < (len(features_1) - 1):
                    j += 1
                else:
                    i += 1
                    j = i + 1
        elif difference[i] < 0 and difference[j] > 0:
            if -difference[i] < difference[j]:
                flow[j][i] = -difference[i]
                difference[j] += difference[i]
                difference[i] = 0
                i += 1
                j = i + 1
            else:
                flow[i][j] = difference[j]
                difference[i] += difference[j]
                difference[j] = 0
                if j < len(features_1) - 1:
                    j += 1
                else:
                    i += 1
                    j = i + 1
        elif difference[i] == difference[j]:
            i += 1
            j = i + 1
        else:
            if j < len(features_1) - 1:
                j += 1
            else:
                i += 1
                j = i + 1

    # Compute sum of distance times flow
    work = 0
    for p in range(len(features_1)):
        for q in range(len(features_1)):
            work += abs(p - q) * flow[p][q]
    
    # 'Normalize' by dividing by total flow
    total_flow = 0
    for i in range(len(features_1)):
        for j in range(len(features_1)):
            total_flow += flow[i][j]
    emd = work / total_flow

    return emd

# from scipy.stats import wasserstein_distance
# def get_emd(v1,v2):
#     return wasserstein_distance(v1,v2)



def main():
    # Parameters
    import os
    os.chdir("./querying")
    query_path = "Humanoid/m140.obj"
    mesh1_path = "Quadruped/m105.obj"
    mesh2_path = "BuildingNonResidential/m385.obj"
    mesh3_path = "Humanoid/m255.obj"
    features_path = "miniDB/features.csv"

    meshes = load_meshes([query_path, mesh1_path, mesh2_path, mesh3_path])
    features_query, label1 = get_features(features_path, query_path)
    features_1, label2 = get_features(features_path, mesh1_path)
    features_2, label4 = get_features(features_path, mesh2_path)
    features_3, label3 = get_features(features_path, mesh3_path)
    # print(f"Features for shape '{query_path}':", features_1)
    # print(f"Features for shape '{mesh1_path}':", features_2)
    # print(f"Features for shape '{mesh3_path}':", features_3)
    # print(f"Features for shape '{mesh2_path}':", features_4)

    print(f"Earth Mover's Distance between {query_path} and {mesh1_path}: {get_emd(features_query, features_1)}")
    print(f"Earth Mover's Distance between {query_path} and {mesh2_path}: {get_emd(features_query, features_2)}")
    print(f"Earth Mover's Distance between {query_path} and {mesh3_path}: {get_emd(features_query, features_3)}")

    visualize(meshes)


if __name__ == "__main__":
    main()
