def get_surface(verticies, triangles):
    S = 0
    for t in triangles:
        a, b, c = verticies[t]
        S += get_triangle_surface(a - b, a - c)
    return S

def get_triangle_surface(v1, v2):
    Xab, Yab, Zab = v1
    Xac, Yac, Zac = v2
    return (((Yab * Zac - Zab * Yac)**2 + (Zab * Xac - Xab * Zac)**2 + (Xab * Yac - Yab * Xac)**2)**0.5) / 2