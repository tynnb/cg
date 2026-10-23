#include "application.hpp"

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Vertex {
    float position[3];
    float color[3];
};

Vertex makeVertex(float x, float y, float z) {
    Vertex v;
    v.position[0] = x;
    v.position[1] = y;
    v.position[2] = z;

    v.color[0] = x + 0.5f;
    v.color[1] = y + 0.5f;
    v.color[2] = z + 0.5f;
    return v;
}

const Vertex cubeVertices[] = {
    makeVertex(-0.5f, -0.5f, -0.5f),
    makeVertex(0.5f, -0.5f, -0.5f),
    makeVertex(0.5f,  0.5f, -0.5f),
    makeVertex(-0.5f,  0.5f, -0.5f),
    makeVertex(-0.5f, -0.5f,  0.5f),
    makeVertex(0.5f, -0.5f,  0.5f),
    makeVertex(0.5f,  0.5f,  0.5f),
    makeVertex(-0.5f,  0.5f,  0.5f),
};

const uint32_t cubeIndices[] = {
    0, 2, 1,  2, 0, 3,
    4, 5, 6,  6, 7, 4,
    0, 7, 3,  7, 0, 4,
    1, 2, 6,  6, 5, 1,
    0, 1, 5,  5, 4, 0,
    3, 7, 6,  6, 2, 3,
};

namespace application {

bool isOrthographic = false;

glm::vec3 objectPosition = glm::vec3(0.0f);
glm::vec3 objectRotation = glm::vec3(30.0f, 45.0f, 0.0f);
glm::vec3 objectScale = glm::vec3(1.0f);
glm::vec3 objectColor = glm::vec3(1.0f);

glm::vec3 objectPosition2 = glm::vec3(1.5f, 0.0f, 0.0f);
glm::vec3 objectRotation2 = glm::vec3(0.0f, 0.0f, 0.0f);
glm::vec3 objectScale2 = glm::vec3(0.5f);
glm::vec3 objectColor2 = glm::vec3(0.0f, 1.0f, 0.0f);

bool isAnimating = true;
float animationSpeed = 1.0f;
float trajectoryRadius = 2.0f;
float trajectoryHeight = 1.0f;
double animationTime = 0.0;

bool initialize() {
    if (!graphics::internal::createBuffers(
        cubeVertices, 8, sizeof(Vertex), cubeIndices, 36)) {
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

    static double lastTime = 0.0;
    double deltaTime = time - lastTime;
    lastTime = time;

    if (isAnimating) {
        animationTime += deltaTime * animationSpeed;
    }

    float t = (float)animationTime;

    objectPosition.x = trajectoryRadius * sin(t * 1.0f);
    objectPosition.y = trajectoryHeight * sin(t * 2.0f);
    objectPosition.z = trajectoryRadius * sin(t * 0.5f);

    objectRotation.x = glm::degrees(t * 1.0f);
    objectRotation.y = glm::degrees(t * 1.5f);
    objectRotation.z = glm::degrees(t * 2.0f);

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, objectPosition);
    model = glm::rotate(model, glm::radians(objectRotation.x), glm::vec3(1, 0, 0));
    model = glm::rotate(model, glm::radians(objectRotation.y), glm::vec3(0, 1, 0));
    model = glm::rotate(model, glm::radians(objectRotation.z), glm::vec3(0, 0, 1));
    model = glm::scale(model, objectScale);

    glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 3), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

    glm::mat4 proj;
    if (isOrthographic) {
        proj = glm::ortho(-2.0f, 2.0f, -2.0f, 2.0f, -10.0f, 10.0f);
    }
    else {
        proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    }

    glm::mat4 mvp = proj * view * model;
    float color[4] = { objectColor.r, objectColor.g, objectColor.b, 1.0f };
    graphics::internal::updateUniformBuffer(&mvp[0][0], color);

    glm::mat4 model2 = glm::mat4(1.0f);
    model2 = glm::translate(model2, objectPosition2);
    model2 = glm::rotate(model2, glm::radians(objectRotation2.x), glm::vec3(1, 0, 0));
    model2 = glm::rotate(model2, glm::radians(objectRotation2.y), glm::vec3(0, 1, 0));
    model2 = glm::rotate(model2, glm::radians(objectRotation2.z), glm::vec3(0, 0, 1));
    model2 = glm::scale(model2, objectScale2);

    glm::mat4 mvp2 = proj * view * model2;
    float color2[4] = { objectColor2.r, objectColor2.g, objectColor2.b, 1.0f };
    graphics::internal::updateUniformBuffer2(&mvp2[0][0], color2);

    ImGui::Begin("Lab 1 Controls");
    ImGui::Checkbox("Orthographic Projection", &isOrthographic);
    ImGui::SliderFloat3("Position", &objectPosition.x, -3.0f, 3.0f);
    ImGui::SliderFloat3("Rotation", &objectRotation.x, 0.0f, 360.0f);
    ImGui::SliderFloat3("Scale", &objectScale.x, 0.1f, 3.0f);
    ImGui::ColorEdit3("Color", &objectColor.x);

    ImGui::Separator();
    ImGui::Text("Object 2");
    ImGui::SliderFloat3("Position 2", &objectPosition2.x, -3.0f, 3.0f);
    ImGui::SliderFloat3("Rotation 2", &objectRotation2.x, 0.0f, 360.0f);
    ImGui::SliderFloat3("Scale 2", &objectScale2.x, 0.1f, 3.0f);
    ImGui::ColorEdit3("Color 2", &objectColor2.x);

    ImGui::Separator();
    ImGui::Text("Animation");
    ImGui::Checkbox("Animate", &isAnimating);
    ImGui::SliderFloat("Speed", &animationSpeed, 0.1f, 5.0f);
    ImGui::SliderFloat("Radius", &trajectoryRadius, 0.5f, 5.0f);
    ImGui::SliderFloat("Height", &trajectoryHeight, 0.0f, 3.0f);
    ImGui::End();

    ImGui::ShowDemoWindow();
}

void render(const graphics::internal::FrameData& fd) {
	(void)fd;
}

} // namespace application