#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec3 out_normal;
layout(location = 1) out vec3 out_worldPos;
layout(location = 2) out vec2 out_uv;

layout(set = 0, binding = 0) uniform SceneUniforms {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 ambientColor;
    vec4 directionalDirection;
    vec4 directionalColor;
    mat4 lightViewProj;
    int pointLightCount;
    int spotLightCount;
} scene;

layout(set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;
    float shininess;
    vec3 specularColor;
} model;

void main() {
    gl_Position = scene.viewProj * model.model * vec4(in_position, 1.0);

    out_normal = mat3(transpose(inverse(model.model))) * in_normal;

    out_worldPos = vec3(model.model * vec4(in_position, 1.0));

    out_uv = in_uv;
}