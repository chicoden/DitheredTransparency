#version 460
precision mediump float;

struct PointLight {
    vec3 position;
    vec3 color;
};

layout(binding = 0, std430) buffer pointLights {
    int pointLightCount;
    PointLight pointLight[];
};

uniform vec3 cameraPosition;
uniform sampler2D ambientTexture;
uniform sampler2D diffuseTexture;
uniform sampler2D specularTexture;
uniform sampler2D glossTexture;

in VsOut {
    vec3 position;
    vec2 uv;
    vec3 normal;
} vsOut;

out vec4 fragColor;

void main() {
    vec3 position = vsOut.position;
    vec2 uv = vsOut.uv;
    vec3 normal = normalize(vsOut.normal);
    vec3 viewDir = normalize(cameraPosition - position);
    vec3 shade = vec3(0.0);

    for (int i = 0; i < pointLightCount; i++) {
        vec3 ambientColor = texture(ambientTexture, uv).rgb;
        vec3 diffuseColor = texture(diffuseTexture, uv).rgb;
        vec3 specularColor = texture(specularTexture, uv).rgb;
        float gloss = texture(glossTexture, uv).r;

        vec3 lightRelPos = pointLight[i].position - position;
        vec3 lightDir = normalize(lightRelPos);
        vec3 reflectDir = -reflect(lightDir, normal);

        vec3 illum = pointLight[i].color / dot(lightRelPos, lightRelPos);
        float diffuse = max(0.0, dot(lightDir, normal));
        float specular = diffuse == 0.0 ? 0.0 : pow(max(0.0, dot(reflectDir, viewDir)), gloss);
        shade += (ambientColor + diffuseColor * diffuse + specularColor * specular) * illum;
    }

    fragColor = vec4(pow(shade, vec3(1.0 / 2.2)), 1.0);
}