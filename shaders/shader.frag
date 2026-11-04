#version 450 core

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_worldPos;
layout(location = 0) out vec4 out_color;

struct PointLight {
    vec3 position;
    float intensity;
    vec3 color;
    float radius;
};

struct SpotLight {
    vec3 position;
    float intensity;
    vec3 direction;
    float radius;
    vec3 color;
    float innerAngle;
    float outerAngle;
    float pad[3];
};

layout(set = 0, binding = 0) uniform SceneUniforms {
    mat4 viewProj;
    vec4 cameraPos;
    vec4 ambientColor;
    vec4 directionalDirection;
    vec4 directionalColor;
    int pointLightCount;
    int spotLightCount;
} scene;

layout(set = 0, binding = 1) readonly buffer PointLights {
    PointLight pointLights[];
} pointLightsBuffer;

layout(set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight spotLights[];
} spotLightsBuffer;

layout(set = 1, binding = 0) uniform ModelUniforms {
    mat4 model;
    vec3 albedo;
    float shininess;
    vec3 specularColor;
} model;

void main() {
    vec3 N = normalize(in_normal);
    vec3 V = normalize(scene.cameraPos.xyz - in_worldPos);

    vec3 ambient = scene.ambientColor.xyz * model.albedo;
    vec3 finalColor = ambient;

    {
        vec3 L = normalize(-scene.directionalDirection.xyz);
        vec3 H = normalize(L + V);
        float diff = max(dot(N, L), 0.0);
        float spec = pow(max(dot(N, H), 0.0), model.shininess);
        finalColor += scene.directionalColor.xyz * scene.directionalColor.w
                    * (diff * model.albedo + spec * model.specularColor);
    }

    for (int i = 0; i < scene.pointLightCount; i++) {
        PointLight light = pointLightsBuffer.pointLights[i];
        vec3 L = normalize(light.position - in_worldPos);
        vec3 H = normalize(L + V);
        float distance = length(light.position - in_worldPos);

        float attenuation = 10.0 / max(distance * distance, 0.0001);

        float radiusFactor = smoothstep(light.radius, light.radius * 0.5, distance);
        attenuation *= radiusFactor;

        float diff = max(dot(N, L), 0.0);
        float spec = pow(max(dot(N, H), 0.0), model.shininess);
        finalColor += light.color * (diff * model.albedo + spec * model.specularColor)
                    * light.intensity * attenuation;
    }

    for (int i = 0; i < scene.spotLightCount; i++) {
        SpotLight light = spotLightsBuffer.spotLights[i];
        vec3 L = normalize(light.position - in_worldPos);
        vec3 H = normalize(L + V);
        float distance = length(light.position - in_worldPos);

        float attenuation = 10.0 / max(distance * distance, 0.0001);

        float radiusFactor = smoothstep(light.radius, light.radius * 0.5, distance);
        attenuation *= radiusFactor;

        float cosOuter = cos(min(light.outerAngle, light.innerAngle));
        float cosInner = cos(max(light.outerAngle, light.innerAngle));
        float cosAngle = dot(-L, normalize(light.direction));
        float spotFactor = smoothstep(cosOuter, cosInner, cosAngle);

        float diff = max(dot(N, L), 0.0);
        float spec = pow(max(dot(N, H), 0.0), model.shininess);

        finalColor += light.color * (diff * model.albedo + spec * model.specularColor)
                    * light.intensity * attenuation * spotFactor;
    }

    out_color = vec4(finalColor, 1.0);
}