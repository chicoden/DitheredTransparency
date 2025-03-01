#version 460
precision highp float;

uniform mat4 cameraMatrix;
uniform mat4 modelMatrix;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

out VsOut {
    vec3 position;
    vec3 normal;
} vsOut;

void main() {
    vsOut.position = (modelMatrix * vec4(position, 1.0)).xyz;
    vsOut.normal = (inverse(transpose(modelMatrix)) * vec4(normal, 0.0)).xyz;
    gl_Position = cameraMatrix * vec4(vsOut.position, 1.0);
}