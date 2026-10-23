#version 450 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 out_color;

layout(std140, set = 0, binding = 0) uniform GlobalUniforms {
    mat4 matrix;
    vec4 color;
} ubo;

void main() {
    gl_Position = ubo.matrix * vec4(in_position, 1.0);
    out_color = in_color * ubo.color.rgb;
}