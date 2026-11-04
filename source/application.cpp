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
};

Vertex makeVertex(float x, float y, float z, float nx, float ny, float nz) {
    Vertex v;
    v.position[0] = x; v.position[1] = y; v.position[2] = z;
    v.normal[0] = nx; v.normal[1] = ny; v.normal[2] = nz;
    return v;
}

const Vertex cubeVertices[] = {
    makeVertex(-0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f),
    makeVertex(0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f),
    makeVertex(0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f),
    makeVertex(-0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f),

    makeVertex(-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f),
    makeVertex(0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f),
    makeVertex(0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f),
    makeVertex(-0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f),

    makeVertex(-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f),
    makeVertex(-0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f),
    makeVertex(-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f),
    makeVertex(-0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f),

    makeVertex(0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f),
    makeVertex(0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f),
    makeVertex(0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f),
    makeVertex(0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f),

    makeVertex(-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f),
    makeVertex(0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f),
    makeVertex(0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f),
    makeVertex(-0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f),

    makeVertex(-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f),
    makeVertex(0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f),
    makeVertex(0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f),
    makeVertex(-0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f),
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

    glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
    float cameraYaw = -90.0f;
    float cameraPitch = 0.0f;
    float cameraFov = 45.0f;

    glm::vec3 objectPosition = glm::vec3(0.0f);
    glm::vec3 objectRotation = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 objectScale = glm::vec3(1.0f);

    glm::vec3 ambientColor = glm::vec3(0.1f);

    glm::vec3 directionalDirection = glm::vec3(-0.5f, -1.0f, -0.5f);
    glm::vec3 directionalColor     = glm::vec3(1.0f, 1.0f, 1.0f);
    float     directionalIntensity = 1.0f;

    glm::vec3 albedo = glm::vec3(0.8f, 0.8f, 0.8f);
    glm::vec3 specularColor = glm::vec3(1.0f, 1.0f, 1.0f);
    float shininess = 32.0f;

    std::vector<graphics::internal::PointLight> pointLights = {
        { {2.0f, 2.0f, 2.0f}, 1.0f, {1.0f, 0.0f, 0.0f}, 2.0f },
        { {3.0f, 0.5f, 2.0f}, 5.0f, {0.0f, 1.0f, 0.0f}, 3.0f },
    };

    std::vector<graphics::internal::SpotLight> spotLights = {
        {
            {0.0f, 3.0f, 3.0f}, 5.0f, {0.0f, -1.0f, -1.0f}, 3.0f,
            {1.0f, 1.0f, 0.0f},
            glm::radians(10.0f), glm::radians(15.0f), {0.0f, 0.0f}
        },
    };

    bool initialize() {
        if (!graphics::internal::createBuffers(
            cubeVertices, 24, sizeof(Vertex), cubeIndices, 36)) {
            return false;
        }
        if (!graphics::internal::createUniformBuffer()) {
            return false;
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

        float viewProjArray[16];
        glm::mat4 viewProj = proj * view;
        memcpy(viewProjArray, glm::value_ptr(viewProj), sizeof(viewProjArray));

        float camPosArray[3] = { cameraPos.x, cameraPos.y, cameraPos.z };
        float ambientArray[3] = { ambientColor.r, ambientColor.g, ambientColor.b };

        int pointLightCount = (int)pointLights.size();
        int spotLightCount = (int)spotLights.size();

        float dirArray[3]    = { directionalDirection.x, directionalDirection.y, directionalDirection.z };
        float dirColArray[3] = { directionalColor.r, directionalColor.g, directionalColor.b };

        graphics::internal::updateSceneUniformBuffer(
            viewProjArray, camPosArray, pointLightCount, ambientArray, spotLightCount,
            dirArray, dirColArray, directionalIntensity);

        float modelArray[16];
        memcpy(modelArray, glm::value_ptr(model), sizeof(modelArray));
        float albedoArray[3] = { albedo.r, albedo.g, albedo.b };
        float specColorArray[3] = { specularColor.r, specularColor.g, specularColor.b };
        graphics::internal::updateModelUniformBuffer(modelArray, albedoArray, shininess, specColorArray);

        graphics::internal::updateLights(pointLights.data(), pointLightCount);
        graphics::internal::updateSpotLights(spotLights.data(), spotLightCount);

        ImGui::Begin("Lab 2: Lighting");

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Cam Pos", &cameraPos.x, -10.0f, 10.0f);
            ImGui::SliderFloat("Yaw", &cameraYaw, -180.0f, 180.0f);
            ImGui::SliderFloat("Pitch", &cameraPitch, -89.0f, 89.0f);
            ImGui::SliderFloat("FOV", &cameraFov, 1.0f, 120.0f);
            ImGui::Checkbox("Orthographic", &isOrthographic);
        }

        if (ImGui::CollapsingHeader("Object", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Position", &objectPosition.x, -5.0f, 5.0f);
            ImGui::SliderFloat3("Rotation", &objectRotation.x, 0.0f, 360.0f);
            ImGui::SliderFloat3("Scale", &objectScale.x, 0.1f, 3.0f);
        }

        if (ImGui::CollapsingHeader("Ambient", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Ambient Color", &ambientColor.x);
        }

        if (ImGui::CollapsingHeader("Directional Light", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat3("Direction", &directionalDirection.x, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", &directionalColor.x);
            ImGui::SliderFloat("Intensity", &directionalIntensity, 0.0f, 5.0f);
        }

        if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Albedo", &albedo.x);
            ImGui::ColorEdit3("Specular", &specularColor.x);
            ImGui::SliderFloat("Shininess", &shininess, 1.0f, 256.0f);
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
                    pointLights.push_back({ {0.0f, 2.0f, 0.0f}, 1.0f, {0.0f, 1.0f, 0.0f}, 2.0f });
                }
            }
        }

        if (ImGui::CollapsingHeader("Spot Lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            for (size_t i = 0; i < spotLights.size(); i++) {
                ImGui::PushID((int)i + 1000);
                if (ImGui::TreeNode(("Spot " + std::to_string(i)).c_str())) {
                    ImGui::SliderFloat3("Position", &spotLights[i].position[0], -5.0f, 5.0f);
                    ImGui::SliderFloat3("Direction", &spotLights[i].direction[0], -1.0f, 1.0f);
                    ImGui::SliderFloat("Intensity", &spotLights[i].intensity, 0.0f, 5.0f);
                    ImGui::ColorEdit3("Color", &spotLights[i].color[0]);
                    ImGui::SliderFloat("Radius", &spotLights[i].radius, 0.1f, 5.0f);
                    ImGui::SliderAngle("Inner Angle", &spotLights[i].innerAngle, 0.0f, 90.0f);
                    ImGui::SliderAngle("Outer Angle", &spotLights[i].outerAngle, 0.0f, 90.0f);
                    if (spotLights[i].outerAngle < spotLights[i].innerAngle) {
                        spotLights[i].outerAngle = spotLights[i].innerAngle;
                    }
                    if (ImGui::Button("Delete")) {
                        spotLights.erase(spotLights.begin() + i);
                        ImGui::TreePop();
                        ImGui::PopID();
                        break;
                    }
                    ImGui::TreePop();
                }
                ImGui::PopID();
            }
            if (ImGui::Button("Add Spot Light")) {
                if (spotLights.size() < 16) {
                    spotLights.push_back({
                        {0.0f, 3.0f, 0.0f}, 1.0f,
                        {0.0f, -1.0f, 0.0f}, 3.0f,
                        {1.0f, 1.0f, 1.0f},
                        glm::radians(15.0f), glm::radians(25.0f), {0.0f, 0.0f}
                        });
                }
            }
        }

        ImGui::End();

        ImGui::ShowDemoWindow();
    }

    void render(const graphics::internal::FrameData& fd) {
        (void)fd;
    }

} // namespace application