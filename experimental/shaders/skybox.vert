#version 460
precision highp float;

uniform float focalLength;
uniform float aspectRatio;

out vec2 coord;

void main() {
    gl_Position = vec4(vec2(gl_VertexID % 2, gl_VertexID / 2) * 2.0 - 1.0, 0.0, 1.0);
    coord = gl_Position.xy / focalLength;
    coord.x *= aspectRatio;
}