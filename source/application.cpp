#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include "application.hpp"

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>
#include <string>

struct Vertex {
    float position[3];
    float normal[3];
    float uv[2];
};

Vertex makeVertex(
    float x, float y, float z,
    float nx, float ny, float nz,
    float u, float v
) {
    Vertex vertex;
    vertex.position[0] = x;
    vertex.position[1] = y;
    vertex.position[2] = z;
    vertex.normal[0] = nx;
    vertex.normal[1] = ny;
    vertex.normal[2] = nz;
    vertex.uv[0] = u;
    vertex.uv[1] = v;
    return vertex;
}

const Vertex cubeVertices[] = {
    makeVertex(-0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f),
    makeVertex( 0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f),
    makeVertex( 0.5f,  0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f),
    makeVertex(-0.5f,  0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f),

    makeVertex(-0.5f, -0.5f,  0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f),
    makeVertex( 0.5f, -0.5f,  0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f),
    makeVertex( 0.5f,  0.5f,  0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f),
    makeVertex(-0.5f,  0.5f,  0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f),

    makeVertex(-0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f),
    makeVertex(-0.5f, -0.5f,  0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f),
    makeVertex(-0.5f,  0.5f,  0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f),
    makeVertex(-0.5f,  0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f),

    makeVertex(0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f),
    makeVertex(0.5f, -0.5f,  0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f),
    makeVertex(0.5f,  0.5f,  0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f),
    makeVertex(0.5f,  0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f),

    makeVertex(-0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f),
    makeVertex( 0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f),
    makeVertex( 0.5f, -0.5f,  0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f),
    makeVertex(-0.5f, -0.5f,  0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f),

    makeVertex(-0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f),
    makeVertex( 0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f),
    makeVertex( 0.5f, 0.5f,  0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f),
    makeVertex(-0.5f, 0.5f,  0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f),
};

const uint32_t cubeIndices[] = {
    0, 1, 2,  2, 3, 0,
    4, 5, 6,  6, 7, 4,
    8, 9, 10, 10, 11, 8,
    12, 13, 14, 14, 15, 12,
    16, 17, 18, 18, 19, 16,
    20, 21, 22, 22, 23, 20
};

namespace application {

    bool isOrthographic = false;
    bool enableWeirdSampling = true;

    float currentTime = 0.0f;

    glm::vec3 cameraPos = glm::vec3(0.0f, 1.5f, 4.0f);
    float cameraYaw = -90.0f;
    float cameraPitch = -15.0f;
    float cameraFov = 45.0f;

    glm::vec3 objectPosition = glm::vec3(0.0f);
    glm::vec3 objectRotation = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 objectScale = glm::vec3(1.0f);

    glm::vec3 ambientColor = glm::vec3(0.1f);

    glm::vec3 directionalDirection = glm::vec3(-0.5f, -1.0f, -0.5f);
    glm::vec3 directionalColor = glm::vec3(1.0f, 1.0f, 1.0f);
    float directionalIntensity = 1.5f;

    glm::vec3 albedo = glm::vec3(0.8f, 0.8f, 0.8f);
    glm::vec3 specularColor = glm::vec3(1.0f, 1.0f, 1.0f);
    float shininess = 32.0f;

    float shadowOrthoSize = 10.0f;
    float shadowNear = 1.0f;
    float shadowFar = 50.0f;
    float shadowDistance = 20.0f;
    float depthBiasConst = 25.0f;
    float depthBiasSlope = 2.0f;
    float shadowBias = 0.002f;

    std::vector<graphics::internal::PointLight> pointLights = {
        { {2.0f, 3.0f, 2.0f}, 0.5f, {1.0f, 1.0f, 1.0f}, 5.0f },
    };

    std::vector<graphics::internal::SpotLight> spotLights = {

    };

    std::vector<std::string> texturePaths = {
        "textures/brick.jpg",
        "textures/wood.jpg",
        "textures/metal.jpg",
        "textures/checker.jpg"
    };
    std::vector<std::string> textureNames = {
        "Brick", "Wood", "Metal", "Checker"
    };
    int selectedTextureIndex = 0;

    bool initialize() {
        if (!graphics::internal::createBuffers(
            cubeVertices, 24, sizeof(Vertex), cubeIndices, 36)) {
            return false;
        }
        if (!graphics::internal::createUniformBuffer()) {
            return false;
        }
        for (const auto& path : texturePaths) {
            if (!graphics::internal::createTextureFromFile(path.c_str())) {
                return false;
            }
        }
        return graphics::internal::createPipeline();
    }

    void shutdown() {
        auto& context = graphics::internal::context;
        vkQueueWaitIdle(context.graphics_queue);
    }

    void update(double time) {
        auto& context = graphics::internal::context;

        float aspect = (float)context.swapchain_extent.width / (float)context.swapchain_extent.height;

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, objectPosition);
        model = glm::rotate(model, glm::radians(objectRotation.x), glm::vec3(1, 0, 0));
        model = glm::rotate(model, glm::radians(objectRotation.y), glm::vec3(0, 1, 0));
        model = glm::rotate(model, glm::radians(objectRotation.z), glm::vec3(0, 0, 1));
        model = glm::scale(model, objectScale);

        glm::vec3 cameraFront;
        static double lastTime = 0.0;
        float dt = static_cast<float>(time - lastTime);
        lastTime = time;
        if (dt > 0.1f) dt = 0.1f;

        ImGuiIO& io = ImGui::GetIO();

        if (!io.WantCaptureMouse && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
            const float sensitivity = 0.15f;
            cameraYaw   += io.MouseDelta.x * sensitivity;
            cameraPitch -= io.MouseDelta.y * sensitivity;
            cameraPitch  = glm::clamp(cameraPitch, -89.9f, 89.9f);

            if (cameraYaw >  180.0f) cameraYaw -= 360.0f;
            if (cameraYaw < -180.0f) cameraYaw += 360.0f;
        }

        glm::vec3 front;
        front.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
        front.y = sin(glm::radians(cameraPitch));
        front.z = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
        front = glm::normalize(front);
        glm::vec3 right = glm::normalize(glm::cross(front, glm::vec3(0, 1, 0)));

        if (!io.WantTextInput) {
            float speed = 3.0f * dt;
            if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) speed *= 3.0f;

            if (ImGui::IsKeyDown(ImGuiKey_W)) cameraPos += front * speed;
            if (ImGui::IsKeyDown(ImGuiKey_S)) cameraPos -= front * speed;
            if (ImGui::IsKeyDown(ImGuiKey_D)) cameraPos += right * speed;
            if (ImGui::IsKeyDown(ImGuiKey_A)) cameraPos -= right * speed;
            if (ImGui::IsKeyDown(ImGuiKey_E)) cameraPos.y += speed;
            if (ImGui::IsKeyDown(ImGuiKey_Q)) cameraPos.y -= speed;
        }
        cameraFront.x = cos(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
        cameraFront.y = sin(glm::radians(cameraPitch));
        cameraFront.z = sin(glm::radians(cameraYaw)) * cos(glm::radians(cameraPitch));
        cameraFront = glm::normalize(cameraFront);

        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, glm::vec3(0, 1, 0));

        glm::mat4 proj;
        if (isOrthographic) {
            proj = glm::ortho(-2.0f, 2.0f, -2.0f, 2.0f, -10.0f, 10.0f);
        }
        else {
            proj = glm::perspective(glm::radians(cameraFov), aspect, 0.1f, 100.0f);
        }
        proj[1][1] *= -1.0f;

        float viewProjArray[16];
        glm::mat4 viewProj = proj * view;
        memcpy(viewProjArray, glm::value_ptr(viewProj), sizeof(viewProjArray));

        float camPosArray[3] = { cameraPos.x, cameraPos.y, cameraPos.z };
        float ambientArray[3] = { ambientColor.r, ambientColor.g, ambientColor.b };

        int pointLightCount = (int)pointLights.size();
        int spotLightCount = (int)spotLights.size();

        float dirArray[3]    = { directionalDirection.x, directionalDirection.y, directionalDirection.z };
        float dirColArray[3] = { directionalColor.r, directionalColor.g, directionalColor.b };

        glm::vec3 lightDir = directionalDirection;
        if (glm::length(lightDir) < 1e-4f) {
            lightDir = glm::vec3(0.0f, -1.0f, 0.0f);
        }
        lightDir = glm::normalize(lightDir);

        const float sceneRadius = shadowOrthoSize + 2.0f;
        const float lo = shadowNear + sceneRadius;
        const float hi = glm::max(lo, shadowFar - sceneRadius);
        float lightDistance = glm::clamp(shadowDistance, lo, hi);

        glm::vec3 lightPos = -lightDir * shadowDistance;
        glm::vec3 up = (glm::abs(lightDir.y) > 0.99f)
                    ? glm::vec3(0.0f, 0.0f, 1.0f)
                    : glm::vec3(0.0f, 1.0f, 0.0f);

        const float nearP = shadowNear;
        const float farP  = glm::max(shadowFar, nearP + 0.1f);

        glm::mat4 lightView = glm::lookAt(lightPos, glm::vec3(0.0f), up);
        glm::mat4 lightProj = glm::ortho(
            -shadowOrthoSize, shadowOrthoSize,
            -shadowOrthoSize, shadowOrthoSize,
            nearP, farP);
        glm::mat4 lightViewProj = lightProj * lightView;

        float lightVP[16];
        memcpy(lightVP, glm::value_ptr(lightViewProj), sizeof(lightVP));

        graphics::internal::updateSceneUniformBuffer(
            viewProjArray, camPosArray, pointLightCount, ambientArray, spotLightCount,
            dirArray, dirColArray, directionalIntensity, lightVP);

        float modelArray[16];
        memcpy(modelArray, glm::value_ptr(model), sizeof(modelArray));
        float albedoArray[3] = { albedo.r, albedo.g, albedo.b };
        float specColorArray[3] = { specularColor.r, specularColor.g, specularColor.b };
        uint32_t modelSlot = 0;

        graphics::internal::updateModelUniformBuffer(modelSlot, modelArray, albedoArray, shininess, specColorArray, 1);

        glm::mat4 floorModel = glm::mat4(1.0f);
        floorModel = glm::translate(floorModel, glm::vec3(0.0f, -0.55f, 0.0f));
        floorModel = glm::scale(floorModel, glm::vec3(15.0f, 0.1f, 15.0f));

        float floorModelArray[16];
        memcpy(floorModelArray, glm::value_ptr(floorModel), sizeof(floorModelArray));

        float floorAlbedoArray[3] = { 0.3f, 0.3f, 0.3f }; 

        graphics::internal::updateModelUniformBuffer(
            1, floorModelArray, floorAlbedoArray, shininess, specColorArray, 0);

        graphics::internal::setDrawModelCount(2);

        graphics::internal::updateLights(pointLights.data(), pointLightCount);
        graphics::internal::updateSpotLights(spotLights.data(), spotLightCount);

        ImGui::Begin("Lab 4: Shadow Mapping");

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Cam Pos", &cameraPos.x, -20.0f, 20.0f);
            ImGui::SliderFloat("Yaw", &cameraYaw, -180.0f, 180.0f);
            ImGui::SliderFloat("Pitch", &cameraPitch, -89.9f, 89.9f);
            ImGui::SliderFloat("FOV", &cameraFov, 1.0f, 120.0f);
            ImGui::Checkbox("Orthographic", &isOrthographic);
        }

        if (ImGui::CollapsingHeader("Object", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Position", &objectPosition.x, -5.0f, 5.0f);
            ImGui::SliderFloat3("Rotation", &objectRotation.x, 0.0f, 360.0f);
            ImGui::SliderFloat3("Scale", &objectScale.x, 0.1f, 3.0f);
            if (ImGui::Button("Reset Rotation")) {
                objectRotation = glm::vec3(0.0f);
            }
        }

        if (ImGui::CollapsingHeader("Ambient", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Ambient Color", &ambientColor.x);
        }

        if (ImGui::CollapsingHeader("Directional Light", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Direction", &directionalDirection.x, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", &directionalColor.x);
            ImGui::SliderFloat("Intensity", &directionalIntensity, 0.0f, 5.0f);
        }

        if (ImGui::CollapsingHeader("Shadow Map", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Ortho Size", &shadowOrthoSize, 1.0f, 30.0f);
            ImGui::SliderFloat("Near", &shadowNear, 0.1f, 60.0f);
            ImGui::SliderFloat("Far", &shadowFar, 1.0f, 200.0f);
            ImGui::SliderFloat("Light Dist", &shadowDistance, 1.0f, 100.0f);
            ImGui::SliderFloat("Depth Bias Const", &depthBiasConst, 0.0f, 500.0f);
            ImGui::SliderFloat("Depth Bias Slope", &depthBiasSlope, 0.0f, 20.0f);
            ImGui::SliderFloat("Shadow Bias", &shadowBias, 0.0f, 0.05f, "%.4f");
        }

        if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Albedo Tint", &albedo.x);
            ImGui::ColorEdit3("Specular", &specularColor.x);
            ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f);
            ImGui::Checkbox("Weird Sampling", &enableWeirdSampling);

            if (ImGui::BeginCombo("Texture", textureNames[selectedTextureIndex].c_str())) {
                for (int i = 0; i < (int)textureNames.size(); i++) {
                    bool isSelected = (selectedTextureIndex == i);
                    if (ImGui::Selectable(textureNames[i].c_str(), isSelected)) {
                        selectedTextureIndex = i;
                        graphics::internal::setActiveTexture(i);
                    }
                    if (isSelected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        }

        if (ImGui::CollapsingHeader("Point Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (size_t i = 0; i < pointLights.size(); i++) {
                ImGui::PushID((int)i);
                if (ImGui::TreeNode(("Light " + std::to_string(i)).c_str())) {
                    ImGui::SliderFloat3("Position", &pointLights[i].position[0], -5.0f, 5.0f);
                    ImGui::SliderFloat("Intensity", &pointLights[i].intensity, 0.0f, 5.0f);
                    ImGui::ColorEdit3("Color", &pointLights[i].color[0]);
                    ImGui::SliderFloat("Radius", &pointLights[i].radius, 0.1f, 5.0f);
                    if (ImGui::Button("Delete")) {
                        pointLights.erase(pointLights.begin() + i);
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
            if (ImGui::Button("Add Light")) {
                if (pointLights.size() < 16) {
                    pointLights.push_back({ {0.0f, 2.0f, 0.0f}, 1.0f, {1.0f, 1.0f, 1.0f}, 2.0f });
                }
            }
        }

        application::currentTime = static_cast<float>(time);
        graphics::internal::setPushConstants(application::currentTime, enableWeirdSampling ? 1 : 0);
        graphics::internal::setShadowParams(lightVP, depthBiasConst, depthBiasSlope, shadowBias);
        
        ImGui::End();
        ImGui::ShowDemoWindow();
    }
} // namespace application