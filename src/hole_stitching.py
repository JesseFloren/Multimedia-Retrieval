import numpy as np

# Hole Stitching
def detect_hole_edges(mesh):
    triangles = np.asarray(mesh.triangles)
    edges = []
    for t1, t2, t3 in triangles:
        v1 = [t1, t2]; v2 = [t2, t3]; v3 = [t3, t1]
        v1.sort(); v2.sort(); v3.sort()

        if v1 in edges: edges.remove(v1)
        else: edges.append(v1)
        
        if v2 in edges: edges.remove(v2)
        else: edges.append(v2)

        if v3 in edges:  edges.remove(v3)
        else: edges.append(v3)
    
    return edges


def get_hole_boundries(edges):
    bounds = []
    while len(edges) > 0:
        bound = [edges[0]]
        edges.remove(edges[0])
        added = True
        while added:
            added = False
            for b in bound:
                for e in edges:
                    if b[0] == e[0] or b[1] == e[0] or b[0] == e[1] or b[1] == e[1]:
                        edges.remove(e)
                        bound.append(e)
                        added = True
        bounds.append(bound)
    return bounds

def get_hole_verticies(mesh, bounds):
    vertices = np.asarray(mesh.vertices)
    bounds_verts = []
    bounds_vertsi = []
    for bound in bounds:
        bound_verts = []
        added = []
        for b1, b2 in bound:
            if b1 not in added:
                bound_verts.append(vertices[b1])
                added.append(b1)
            
            if b2 not in added:
                bound_verts.append(vertices[b2])
                added.append(b2)
        
        bounds_vertsi.append(added)
        bounds_verts.append(bound_verts)
    return bounds_verts

def generate_fan_stitch(mesh, bounds, bounds_verts):
    new_vertices = np.asarray(mesh.vertices)
    new_triangles = np.asarray(mesh.triangles)

    for i in range(len(bounds)):
        barycenter = np.mean(bounds_verts[i], axis=0)
        bcenter = len(new_vertices)
        new_vertices = np.concatenate([new_vertices, [barycenter]])
        for e1, e2 in bounds[i]:
            new_triangles = np.concatenate([new_triangles, [[e2, e1, bcenter]]])

    return new_vertices, new_triangles

def stitch_mesh_holes(mesh):
    edges = detect_hole_edges(mesh)
    bounds = get_hole_boundries(edges)
    hole_verts = get_hole_verticies(mesh, bounds)
    return generate_fan_stitch(mesh, bounds, hole_verts)