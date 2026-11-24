#version 450 core

layout(location = 0) in vec3 in_position;

layout(set = 0, binding = 0) uniform ModelUniforms {
    mat4 model;
} model;

layout(push_constant) uniform PushConstants {
    mat4 lightViewProj;
} pc;

void main() {
    gl_Position = pc.lightViewProj * model.model * vec4(in_position, 1.0);
}