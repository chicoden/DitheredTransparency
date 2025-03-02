import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders
from obj_loader import load_obj
from PIL import Image
import numpy as np
import ctypes

class PointLight(ctypes.Structure):
    _fields_ = [
        ("position", ctypes.c_float * 3),
        ("pad1", ctypes.c_float),
        ("color", ctypes.c_float * 3),
        ("pad2", ctypes.c_float)
    ]

class PointLights(ctypes.Structure):
    _fields_ = [
        ("pointLightCount", ctypes.c_int32),
        ("pad1", ctypes.c_int32 * 3),
        ("pointLight", PointLight * 1)
    ]

def perspective_proj(focal_length, aspect_ratio, z_near, z_far):
    z_scale = z_far / (z_near - z_far)
    return np.array([
        [focal_length / aspect_ratio, 0.0, 0.0, 0.0],
        [0.0, focal_length, 0.0, 0.0],
        [0.0, 0.0, z_scale, z_scale * z_near],
        [0.0, 0.0, -1.0, 0.0]
    ], dtype=np.float32)

def make_image_texture(image):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    data = np.array(image.convert("RGB").transpose(Image.FLIP_TOP_BOTTOM))
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, *image.size, 0, GL_RGB, GL_UNSIGNED_BYTE, data)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    return tex

def make_color_texture(color):
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    data = np.array(color, dtype=np.float32)
    if len(color) == 3:
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 1, 1, 0, GL_RGB, GL_FLOAT, data)

    else: # len(color) == 1
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1, 1, 0, GL_RED, GL_FLOAT, data)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    return tex

class Material:
    def __init__(self, descriptor):
        if descriptor["map_Ka"]:
            self.ambient = make_image_texture(descriptor["map_Ka"])

        else:
            self.ambient = make_color_texture(descriptor["Ka"] or [0.0, 0.0, 0.0])

        if descriptor["map_Kd"]:
            self.diffuse = make_image_texture(descriptor["map_Kd"])

        else:
            self.diffuse = make_color_texture(descriptor["Kd"] or [0.0, 0.0, 0.0])

        if descriptor["map_Ks"]:
            self.specular = make_image_texture(descriptor["map_Ks"])

        else:
            self.specular = make_color_texture(descriptor["Ks"] or [0.0, 0.0, 0.0])

        if descriptor["map_Ns"]:
            raise NotImplementedError("check me out")

        else:
            self.gloss = make_color_texture([descriptor["Ns"] or 0.0])

pygame.init()

resolution = (800, 600)
pygame.display.set_mode(resolution, pygame.DOUBLEBUF | pygame.OPENGL)
pygame.display.set_caption("Dithered Transparency")
glClearColor(0.0, 0.0, 0.0, 1.0)

meshes = [
    [*load_obj(r"C:\Users\Pytho\3DModels\Shrek\shrek.obj"), np.eye(4, dtype=np.float32)]
]

for mesh in meshes:
    vertex_buffer, index_buffer = glGenBuffers(2)

    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
    glBufferData(GL_ARRAY_BUFFER, mesh[0], GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh[1], GL_STATIC_DRAW)

    mesh[0] = vertex_buffer
    mesh[1] = index_buffer

    for mtl_group in mesh[2]:
        mtl_group[0] = Material(mtl_group[0])
        mtl_group[1], mtl_group[2] = mtl_group[1] * 12, (mtl_group[2] - mtl_group[1]) * 3

with (
    open("shaders/simple_transform.vert", "r") as vs_source,
    open("shaders/phong_light.frag", "r") as fs_source
):
    shader = shaders.compileProgram(
        shaders.compileShader(vs_source.read(), GL_VERTEX_SHADER),
        shaders.compileShader(fs_source.read(), GL_FRAGMENT_SHADER)
    )

camera_mat_loc = glGetUniformLocation(shader, "cameraMatrix")
normal_mat_loc = glGetUniformLocation(shader, "normalMatrix")
model_mat_loc = glGetUniformLocation(shader, "modelMatrix")
camera_pos_loc = glGetUniformLocation(shader, "cameraPosition")
ambient_tex_loc = glGetUniformLocation(shader, "ambientTexture")
diffuse_tex_loc = glGetUniformLocation(shader, "diffuseTexture")
specular_tex_loc = glGetUniformLocation(shader, "specularTexture")
gloss_tex_loc = glGetUniformLocation(shader, "glossTexture")

proj_matrix = perspective_proj(1.0, resolution[0] / resolution[1], 0.1, 10.0)
camera_orientation = np.eye(4, dtype=np.float32)

point_lights = PointLights()
point_lights.pointLightCount = 1
point_lights.pointLight[0].position[:] = [-2.0, 2.0, 2.0]
point_lights.pointLight[0].color[:] = [1.0, 1.0, 1.0]

point_lights_buffer = glGenBuffers(1)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, point_lights_buffer)
glBufferData(GL_SHADER_STORAGE_BUFFER, bytes(point_lights), GL_DYNAMIC_DRAW)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    #yaw, pitch = pygame.mouse.get_rel()
    #yaw *= 0.01
    #pitch *= -0.01
    yaw, pitch = 0.0001, -0.00005
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    camera_orientation = np.dot(
        camera_orientation,
        np.dot(
            np.array([
                [+cy, 0.0, -sy, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [+sy, 0.0, +cy, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32),
            np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, +cp, -sp, 0.0],
                [0.0, +sp, +cp, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)
        )
    )

    camera_matrix = np.dot(
        camera_orientation,
        np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
    )

    point_lights.pointLight[0].position[:] = camera_matrix[0:3, 3]
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, point_lights_buffer)
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes(point_lights))

    glViewport(0, 0, *resolution)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(shader)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, point_lights_buffer)
    glUniformMatrix4fv(camera_mat_loc, 1, GL_TRUE, np.dot(proj_matrix, np.linalg.inv(camera_matrix)))
    glUniform3f(camera_pos_loc, *camera_matrix[0:3, 3])

    for vertex_buffer, index_buffer, mtl_groups, model_matrix in meshes:
        glUniformMatrix4fv(normal_mat_loc, 1, GL_TRUE, np.transpose(np.linalg.inv(model_matrix)))
        glUniformMatrix4fv(model_mat_loc, 1, GL_TRUE, model_matrix)

        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer)

        glEnableVertexAttribArray(0) # Vertices
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, GLvoidp(0))

        glEnableVertexAttribArray(1) # UVs
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, GLvoidp(12))

        glEnableVertexAttribArray(2) # Normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, GLvoidp(20))

        for material, offset, element_count in mtl_groups:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, material.ambient)
            glUniform1i(ambient_tex_loc, 0)

            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, material.diffuse)
            glUniform1i(diffuse_tex_loc, 1)

            glActiveTexture(GL_TEXTURE2)
            glBindTexture(GL_TEXTURE_2D, material.specular)
            glUniform1i(specular_tex_loc, 2)

            glActiveTexture(GL_TEXTURE3)
            glBindTexture(GL_TEXTURE_2D, material.gloss)
            glUniform1i(gloss_tex_loc, 3)

            glDrawElements(GL_TRIANGLES, element_count, GL_UNSIGNED_INT, GLvoidp(offset))

    pygame.display.flip()
