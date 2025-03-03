import pygame
from OpenGL.GL import *
from OpenGL.GL import shaders
from obj_loader import load_obj
from PIL import Image
import numpy as np
import ctypes

CAMERA_FOCAL_LENGTH  = 2.0
CAMERA_MOVEMENT_RATE = 0.005
CAMERA_ROTATION_RATE = 4.0
KEY_EXIT_FULLSCREEN  = pygame.K_ESCAPE
KEY_MOVE_FORWARD     = pygame.K_w
KEY_MOVE_LEFT        = pygame.K_a
KEY_MOVE_BACKWARD    = pygame.K_s
KEY_MOVE_RIGHT       = pygame.K_d

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
    glGenerateMipmap(GL_TEXTURE_2D)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
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
        map_Ka = descriptor.get("map_Ka", None)
        if map_Ka:
            self.ambient = make_image_texture(map_Ka)

        else:
            self.ambient = make_color_texture(descriptor.get("Ka", None) or [0.0, 0.0, 0.0])

        map_Kd = descriptor.get("map_Kd", None)
        if map_Kd:
            self.diffuse = make_image_texture(map_Kd)

        else:
            self.diffuse = make_color_texture(descriptor.get("Kd", None) or [0.0, 0.0, 0.0])

        map_Ks = descriptor.get("map_Ks", None)
        if map_Ks:
            self.specular = make_image_texture(map_Ks)

        else:
            self.specular = make_color_texture(descriptor.get("Ks", None) or [0.0, 0.0, 0.0])

        map_Ns = descriptor.get("map_Ns", None)
        if map_Ns:
            raise NotImplementedError("check me out")

        else:
            self.gloss = make_color_texture([descriptor.get("Ns", None) or 0.0])

class Mesh:
    def __init__(self, vertices, indices, faces_by_mtl):
        self.vertex_buffer, self.index_buffer = glGenBuffers(2)

        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)

        self.faces_by_mtl = []
        for mtl_descriptor, start_index, end_index in faces_by_mtl:
            self.faces_by_mtl.append([
                Material(mtl_descriptor),
                start_index * 12,
                (end_index - start_index) * 3
            ])

        self.transform = np.eye(4, dtype=np.float32)

class Light(ctypes.Structure):
    _fields_ = [
        ("direction", ctypes.c_float * 3),
        ("pad1", ctypes.c_float),
        ("emission", ctypes.c_float * 3),
        ("pad2", ctypes.c_float)
    ]

class Lights(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_int32),
        ("pad1", ctypes.c_int32 * 3),
        ("lights", Light * 1)
    ]

class RawMouse:
    MARGIN = 50

    def __init__(self):
        win_size_x, win_size_y = pygame.display.get_window_size()
        self.margin_x_max = win_size_x - RawMouse.MARGIN - 1
        self.margin_y_max = win_size_y - RawMouse.MARGIN - 1
        self.x, self.y = pygame.mouse.get_pos()
        self.adjusted = False

    def handle_motion_event(self, event):
        if self.adjusted:
            self.adjusted = False

        else:
            self.x += event.rel[0]
            self.y += event.rel[1]

            adjusted_x = max(RawMouse.MARGIN, min(self.margin_x_max, event.pos[0]))
            adjusted_y = max(RawMouse.MARGIN, min(self.margin_y_max, event.pos[1]))
            if adjusted_x != event.pos[0] or adjusted_y != event.pos[1]:
                pygame.mouse.set_pos(adjusted_x, adjusted_y)
                self.adjusted = True

pygame.init()

info = pygame.display.Info()
resolution = (info.current_w, info.current_h)
aspect_ratio = resolution[0] / resolution[1]

pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
pygame.display.set_mode(resolution, pygame.NOFRAME | pygame.DOUBLEBUF | pygame.OPENGL)
glClearColor(0.0, 0.0, 0.0, 1.0)

pygame.mouse.set_visible(False)
mouse = RawMouse()

with (
    open("shaders/simple_transform.vert", "r") as vs_source,
    open("shaders/phong_light.frag", "r") as fs_source
):
    simple_shader = shaders.compileProgram(
        shaders.compileShader(vs_source.read(), GL_VERTEX_SHADER),
        shaders.compileShader(fs_source.read(), GL_FRAGMENT_SHADER)
    )

(
    proj_view_mat_loc,
    normal_mat_loc,
    model_mat_loc,
    camera_pos_loc,
    ambient_tex_loc,
    diffuse_tex_loc,
    specular_tex_loc,
    gloss_tex_loc
) = (
    glGetUniformLocation(simple_shader, name) for name in (
        "projViewMatrix",
        "normalMatrix",
        "modelMatrix",
        "cameraPosition",
        "ambientTexture",
        "diffuseTexture",
        "specularTexture",
        "glossTexture"
    )
)

with (
    open("shaders/skybox.vert", "r") as vs_source,
    open("shaders/skybox.frag", "r") as fs_source
):
    skybox_shader = shaders.compileProgram(
        shaders.compileShader(vs_source.read(), GL_VERTEX_SHADER),
        shaders.compileShader(fs_source.read(), GL_FRAGMENT_SHADER)
    )

(
    focal_length_loc,
    aspect_ratio_loc,
    camera_mat_loc,
    env_loc
) = (
    glGetUniformLocation(skybox_shader, name) for name in (
        "focalLength",
        "aspectRatio",
        "cameraMatrix",
        "env"
    )
)

proj_matrix = perspective_proj(CAMERA_FOCAL_LENGTH, aspect_ratio, 0.2, 100.0)
camera_position = np.array([0.0, 1.0, 3.0], dtype=np.float32)
key_states = {
    KEY_MOVE_FORWARD:  False,
    KEY_MOVE_LEFT:     False,
    KEY_MOVE_BACKWARD: False,
    KEY_MOVE_RIGHT:    False
}

meshes = [
    Mesh(*load_obj(r"C:\Users\Pytho\3DModels\TestScene\street_environment.obj")),
    Mesh(*load_obj(r"C:\Users\Pytho\3DModels\Shrek\shrek.obj"))
]

meshes[1].transform = np.array([
    [ 0.0,  0.0, -1.0,  0.0],
    [ 0.0,  1.0,  0.0,  1.1],
    [ 1.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0]
], dtype=np.float32)

for material, offset, element_count in meshes[1].faces_by_mtl:
    glBindTexture(GL_TEXTURE_2D, material.diffuse)

    width = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
    height = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)

    data = np.empty((height, width, 3), dtype=np.uint8)
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
    data = np.asarray(data * 0.2, dtype=np.uint8)

    ambient_tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, ambient_tex)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
    glGenerateMipmap(GL_TEXTURE_2D)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    glDeleteTextures(1, [material.ambient])
    material.ambient = ambient_tex

light_data = Lights()
light_data.count = 1
light_data.lights[0].direction[:] = [-0.69691911, 0.59238124, 0.40421308]
light_data.lights[0].emission[:] = [0.7, 0.7, 0.7]
lights_buffer = glGenBuffers(1)
glBindBuffer(GL_SHADER_STORAGE_BUFFER, lights_buffer)
glBufferData(GL_SHADER_STORAGE_BUFFER, bytes(light_data), GL_DYNAMIC_DRAW)

env_map = glGenTextures(1)
glBindTexture(GL_TEXTURE_CUBE_MAP, env_map)
for i, face in enumerate(("+x", "-x", "+y", "-y", "+z", "-z")):
    data = np.array(Image.open("simple_sky\\" + face + ".jpg"))
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, data.shape[1], data.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, data)

glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

def terminate():
    pygame.mouse.set_visible(True)
    pygame.quit()
    exit()

frame_count = 0
last_dt = 0
while True:
    frame_start = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            terminate()

        elif event.type == pygame.KEYDOWN:
            if event.key in key_states:
                key_states[event.key] = True

            elif event.key == KEY_EXIT_FULLSCREEN:
                terminate()

        elif event.type == pygame.KEYUP:
            if event.key in key_states:
                key_states[event.key] = False

        elif event.type == pygame.MOUSEMOTION:
            mouse.handle_motion_event(event)

    yaw = (mouse.x - resolution[0] / 2) / resolution[1] * CAMERA_ROTATION_RATE
    pitch = (resolution[1] / 2 - mouse.y) / resolution[1] * CAMERA_ROTATION_RATE
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    camera_matrix = np.dot(
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

    camera_matrix[0:3, 3] = camera_position

    x_dir = camera_matrix[0:3, 0].copy()
    y_dir = x_dir.copy()
    y_dir[0], y_dir[2] = -y_dir[2], y_dir[0]
    move_x = key_states[KEY_MOVE_RIGHT  ] - key_states[KEY_MOVE_LEFT    ]
    move_y = key_states[KEY_MOVE_FORWARD] - key_states[KEY_MOVE_BACKWARD]
    camera_position += (move_x * x_dir - move_y * y_dir) * CAMERA_MOVEMENT_RATE * last_dt

    #glBindBuffer(GL_SHADER_STORAGE_BUFFER, lights_buffer)
    #glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes(light_data))

    glViewport(0, 0, *resolution)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glUseProgram(simple_shader)

    glEnable(GL_DEPTH_TEST)
    #glEnable(GL_CULL_FACE)
    #glCullFace(GL_BACK)

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, lights_buffer)
    glUniformMatrix4fv(proj_view_mat_loc, 1, GL_TRUE, np.dot(proj_matrix, np.linalg.inv(camera_matrix)))
    glUniform3f(camera_pos_loc, *camera_position)

    for mesh in meshes:
        glUniformMatrix4fv(normal_mat_loc, 1, GL_TRUE, np.transpose(np.linalg.inv(mesh.transform)))
        glUniformMatrix4fv(model_mat_loc, 1, GL_TRUE, mesh.transform)

        glBindBuffer(GL_ARRAY_BUFFER, mesh.vertex_buffer)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.index_buffer)

        glEnableVertexAttribArray(0) # Vertices
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, GLvoidp(0))

        glEnableVertexAttribArray(1) # UVs
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, GLvoidp(12))

        glEnableVertexAttribArray(2) # Normals
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, GLvoidp(20))

        for material, offset, element_count in mesh.faces_by_mtl:
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

    glUseProgram(skybox_shader)

    glUniform1f(focal_length_loc, CAMERA_FOCAL_LENGTH)
    glUniform1f(aspect_ratio_loc, aspect_ratio)
    glUniformMatrix4fv(camera_mat_loc, 1, GL_TRUE, camera_matrix)
    glUniform1i(env_loc, 0)

    glDepthFunc(GL_LEQUAL)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    glDepthFunc(GL_LESS)

    pygame.display.flip()

    frame_end = pygame.time.get_ticks()
    last_dt = frame_end - frame_start

    if frame_count % 1000 == 0:
        print(frame_end - frame_start)

    frame_count += 1
