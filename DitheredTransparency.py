import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders
from PIL import Image
import numpy as np
import os

resolution = (800, 600)
pygame.init()
pygame.display.set_mode(resolution, pygame.DOUBLEBUF | pygame.OPENGL)
pygame.display.set_caption("Dithered Transparency")
glClearColor(0.0, 0.0, 0.0, 1.0)

glEnable(GL_DEPTH_TEST)
glDisable(GL_CULL_FACE)

def make_texture(color_or_image):
    return color_or_image###

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    if type(color_or_image) is float:
        color_or_image = [color_or_image]

    if hasattr(color_or_image, "__iter__"):
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, 1, 1, 0, source_format, source_type, np.array(color_or_image, dtype=dtype))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    else:
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, *color_or_image.size, 0, source_format, source_type, np.array(color_or_image))
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    return tex

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
                cur_mtl = {"Ka": None, "Kd": None, "Ks": None, "Ns": None}
                materials[chunks[1]] = cur_mtl

            elif chunks[0] in ("Ka", "Kd", "Ks", "Ns", "map_Ka", "map_Kd", "map_Ks", "map_Ns"):
                cur_mtl[chunks[0][-2:]] = make_texture(
                    Image.open(os.path.join(root, chunks[1])) if chunks[0].startswith("map_")
                    else [float(chunks[1]), float(chunks[2]), float(chunks[3])] if chunks[0][-2] == "K"
                    else float(chunks[1])
                )

        return materials

def load_obj(path, default_mtl):
    with open(path, "r") as file:
        materials = {}
        vertices = []
        uvs = []
        normals = []
        faces = []
        faces_by_mtl = [[default_mtl, 0, -1]]
        for line in file.readlines():
            chunks = line.split()
            if len(chunks) == 0:
                continue

            if chunks[0] == "mtllib":
                root = path.rsplit(os.sep, maxsplit=1)[0]
                materials = load_mtl(os.path.join(root, chunks[1]), root)
                print(materials)###

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
                a, b, c = (np.array(vertices[vi], dtype=np.float64) for vi, ti, ni in face)
                perp = np.cross(b - a, c - a)
                length = np.linalg.norm(perp)
                if length != 0:
                    perp /= length

                face_normal_index = len(normals)
                normals.append([*map(float, perp)])

            for vertex in face:
                if vertex[1] == -1:
                    vertex[1] = null_uv_index

                if vertex[2] == -1:
                    vertex[2] = face_normal_index

        unique_vertices = {tuple(vertex) for face in faces for vertex in face}
        vertex_indices = {vertex: index for index, vertex in enumerate(unique_vertices)}
        attrib_groups = [[*vertices[vi], *uvs[ti], *normals[ni]] for vi, ti, ni in unique_vertices]
        indices = [[vertex_indices[tuple(vertex)] for vertex in face] for face in faces]

        return attrib_groups, indices, faces_by_mtl

ag, i, mg = load_obj(r"C:\Users\Pytho\3DModels\Shrek\shrek.obj", None)###
1/0###

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    glViewport(0, 0, *resolution)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    ###

    pygame.display.flip()
