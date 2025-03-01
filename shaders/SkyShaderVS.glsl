#version 460
precision highp float;

uniform mat4 cameraMatrix;

layout(location = 0) in vec3 position;
out vec3 rayDir;

void main() {
    rayDir = position;
    gl_Position = (cameraMatrix * vec4(position, 0.0)).xyww;
}