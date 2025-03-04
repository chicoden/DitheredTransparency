from PIL import Image
import numpy as np

def compute_normal(a, b, c):
    px, py, pz = (bx - ax for bx, ax in zip(b, a))
    qx, qy, qz = (cx - ax for cx, ax in zip(c, a))
    perp = [
        py * qz - pz * qy,
        pz * qx - px * qz,
        px * qy - py * qx
    ]

    length = np.sqrt(perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2])
    if length != 0:
        perp[0] /= length
        perp[1] /= length
        perp[2] /= length

    return perp

def parse_obj_indices(s):
    indices = [-1, -1, -1]
    for i, chunk in enumerate(s.split("/")):
        if chunk != "":
            indices[i] = int(chunk) - 1

    return indices

def load_mtl(path, root):
    with open(path, "r") as file:
        materials = {}
        cur_mtl = None
        for line in file.readlines():
            chunks = line.split()
            if len(chunks) == 0:
                continue

            if chunks[0] == "newmtl":
                cur_mtl = {}
                materials[chunks[1]] = cur_mtl

            elif chunks[0] in ("Ka", "Kd", "Ks"):
                cur_mtl[chunks[0]] = [float(chunks[1]), float(chunks[2]), float(chunks[3])]

            elif chunks[0] == "Ns":
                cur_mtl[chunks[0]] = float(chunks[1])

            elif chunks[0] in ("map_Ka", "map_Kd", "map_Ks", "map_Ns"):
                cur_mtl[chunks[0]] = Image.open(root + chunks[1])

        return materials

def load_obj(path):
    with open(path, "r") as file:
        materials = {}
        vertices = []
        uvs = []
        normals = []
        faces = []
        faces_by_mtl = [[None, 0, -1]]
        for line in file.readlines():
            chunks = line.split()
            if len(chunks) == 0:
                continue

            if chunks[0] == "mtllib":
                root = path.replace("\\", "/").rsplit("/", maxsplit=1)[0] + "/"
                materials = load_mtl(root + chunks[1], root)

            elif chunks[0] == "usemtl":
                faces_by_mtl[-1][2] = len(faces)
                if faces_by_mtl[-1][1] == faces_by_mtl[-1][2]:
                    faces_by_mtl.pop()

                faces_by_mtl.append([materials[chunks[1]], len(faces), -1])

            elif chunks[0] == "v":
                vertices.append([float(chunks[1]), float(chunks[2]), float(chunks[3])])

            elif chunks[0] == "vt":
                uvs.append([float(chunks[1]), float(chunks[2])])

            elif chunks[0] == "vn":
                normals.append([float(chunks[1]), float(chunks[2]), float(chunks[3])])

            elif chunks[0] == "f":
                face_indices = [parse_obj_indices(chunks[i]) for i in range(1, len(chunks))]
                for i in range(1, len(face_indices) - 1):
                    faces.append([
                        face_indices[0],
                        face_indices[i],
                        face_indices[i + 1]
                    ])

        faces_by_mtl[-1][2] = len(faces)
        if faces_by_mtl[-1][1] == faces_by_mtl[-1][2]:
            faces_by_mtl.pop()

        null_uv_index = -1
        if any(vertex[1] == -1 for face in faces for vertex in face):
            try:
                null_uv_index = uvs.index([0.0, 0.0])

            except ValueError:
                null_uv_index = len(uvs)
                uvs.append([0.0, 0.0])

        for face in faces:
            face_normal_index = -1
            if any(vertex[2] == -1 for vertex in face):
                face_normal_index = len(normals)
                normals.append(compute_normal(vertices[face[0][0]], vertices[face[1][0]], vertices[face[2][0]]))

            for vertex in face:
                if vertex[1] == -1:
                    vertex[1] = null_uv_index

                if vertex[2] == -1:
                    vertex[2] = face_normal_index

        unique_vertices = {tuple(vertex) for face in faces for vertex in face}
        vertex_attribs = np.empty((len(unique_vertices), 8), dtype=np.float32)
        vertex_indices = {}
        for index, vertex in enumerate(unique_vertices):
            vertex_indices[vertex] = index
            vertex_attribs[index] = [*vertices[vertex[0]], *uvs[vertex[1]], *normals[vertex[2]]]

        indices = np.empty((len(faces), 3), dtype=np.uint32)
        for i, face in enumerate(faces):
            for j, vertex in enumerate(face):
                indices[i][j] = vertex_indices[tuple(vertex)]

        return vertex_attribs, indices, faces_by_mtl
