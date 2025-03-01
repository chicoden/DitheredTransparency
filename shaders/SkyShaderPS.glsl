#version 460
precision mediump float;

uniform samplerCube env;

in vec3 rayDir;
out vec4 fragColor;

void main() {
    fragColor = texture(env, rayDir);
}