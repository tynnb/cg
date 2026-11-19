#version 450 core

layout(location = 0) in vec3 in_normal;
layout(location = 1) in vec3 in_worldPos;
layout(location = 2) in vec2 in_uv;
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

layout(set = 2, binding = 0) uniform sampler2D albedoTexture;

layout(push_constant) uniform PushConstants {
    float time;
    int enableWeirdSampling;
} pc;

void main() {
    vec3 N = normalize(in_normal);
    vec3 V = normalize(scene.cameraPos.xyz - in_worldPos);

        vec3 surfaceAlbedo;
        if (pc.enableWeirdSampling != 0) {
            float stepIndex = floor(pc.time);
            float progress = fract(pc.time);
            float a = fract(sin(stepIndex * 127.1) * 43758.5453);
            float b = fract(sin((stepIndex + 1.0) * 127.1) * 43758.5453);
            float noise = mix(a, b, smoothstep(0.0, 1.0, progress));
            float radius = mix(0.02, 0.08, noise);
            vec2 uv1 = in_uv + vec2(sin(pc.time), cos(pc.time)) * radius;
            vec2 uv2 = in_uv * 2.0 + vec2(pc.time * 0.1, 0.0);
            vec3 t1 = texture(albedoTexture, uv1).rgb;
            vec3 t2 = texture(albedoTexture, uv2).rgb;

            surfaceAlbedo = mix(t1, t2, 0.5);

            surfaceAlbedo *= vec3(
                0.8 + 0.2 * sin(pc.time * 2.0),
                0.8 + 0.2 * cos(pc.time * 2.0),
                1.0
            );
        } else {
            surfaceAlbedo = texture(albedoTexture, in_uv).rgb;
        }
        surfaceAlbedo *= model.albedo;

    vec3 ambient = scene.ambientColor.xyz * surfaceAlbedo;
    vec3 finalColor = ambient;

    {
        vec3 L = normalize(-scene.directionalDirection.xyz);
        vec3 H = normalize(L + V);
        float diff = max(dot(N, L), 0.0);
        float spec = pow(max(dot(N, H), 0.0), model.shininess);
        finalColor += scene.directionalColor.xyz * scene.directionalColor.w
                    * (diff * surfaceAlbedo + spec * model.specularColor);
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

        finalColor += light.color * (diff * surfaceAlbedo + spec * model.specularColor)
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

        finalColor += light.color * (diff * surfaceAlbedo + spec * model.specularColor)
                    * light.intensity * attenuation * spotFactor;
    }

    out_color = vec4(finalColor, 1.0);
}