#version 460
precision highp float;

uniform mat4 projViewMatrix;
uniform mat4 normalMatrix;
uniform mat4 modelMatrix;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;

out VsOut {
    vec3 position;
    vec2 uv;
    vec3 normal;
} vsOut;

void main() {
    vsOut.position = (modelMatrix * vec4(position, 1.0)).xyz;
    vsOut.uv = uv;
    vsOut.normal = (normalMatrix * vec4(normal, 0.0)).xyz;
    gl_Position = projViewMatrix * vec4(vsOut.position, 1.0);
}