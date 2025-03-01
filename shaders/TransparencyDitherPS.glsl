#version 460
precision mediump float;

layout(binding = 0) uniform sampler2D bayerPattern;
uniform vec3 cameraPosition;
uniform vec3 lightDirection;
uniform vec3 objColor;
uniform float objOpacity;
uniform int objId;

in VsOut {
    vec3 position;
    vec3 normal;
} vsOut;

out vec4 fragColor;

// https://www.shadertoy.com/view/4djSRW
vec2 hash21(float p)
{
	vec3 p3 = fract(vec3(p) * vec3(.1031, .1030, .0973));
	p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx+p3.yz)*p3.zy);
}

void main() {
    vec2 bayerUv = floor(gl_FragCoord.xy) / textureSize(bayerPattern, 0);
    bayerUv += hash21(float(objId * 2 + int(gl_FrontFacing)));
    float bayerValue = texture(bayerPattern, bayerUv)[0];
    if (objOpacity < bayerValue) discard;

    vec3 normal = normalize(vsOut.normal);
    vec3 viewDir = normalize(cameraPosition - vsOut.position);
    if (dot(normal, viewDir) < 0.0) normal = -normal;

    float diffuse = max(0.0, dot(normalize(lightDirection), normal));
    vec3 color = objColor * diffuse;
    fragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}