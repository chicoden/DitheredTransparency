#version 460
precision mediump float;

uniform mat4 cameraMatrix;
uniform samplerCube env;

in vec2 coord;
out vec4 fragColor;

void main() {
    vec3 rd;
    rd.z = -inversesqrt(dot(coord, coord) + 1.0);
    rd.xy = coord * -rd.z;
    rd = (cameraMatrix * vec4(rd, 0.0)).xyz;
    fragColor = texture(env, rd);
}