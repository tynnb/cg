#pragma once

#include <cstdint>

#include <vulkan/vulkan_core.h>

#include <vk_mem_alloc.h>

struct GLFWwindow;

namespace graphics::internal {

	struct Context {
		VkPhysicalDevice physical_device;
		VkDevice device;

		VmaAllocator allocator;

		VkQueue graphics_queue;
		uint32_t graphics_queue_index;

		VkFormat swapchain_format;
		VkExtent2D swapchain_extent;

		VkRenderPass render_pass;
	};

	struct FrameData {
		VkFramebuffer framebuffer;
		VkCommandBuffer command_buffer;
	};

	struct PointLight {
		float position[3];
		float intensity;
		float color[3];
		float radius;
	};

	struct SpotLight {
		float position[3];
		float intensity;
		float direction[3];
		float radius;
		float color[3];
		float innerAngle;
		float outerAngle;
		float pad[3];   // было pad[2] -> нужно pad[3]
	};

	static_assert(sizeof(SpotLight) == 64, "SpotLight must match std430 layout in shader");
	static_assert(sizeof(PointLight) == 32, "PointLight must match std430 layout in shader");

	extern Context context;

	bool initialize(GLFWwindow* const window);
	void shutdown();

	void resize(uint32_t width, uint32_t height);

	FrameData prepare();
	void submitAndPresent();

	bool createBuffers(const void* vertices, size_t vertex_count, size_t vertex_size,
		const uint32_t* indices, size_t index_count);

	bool createUniformBuffer();

	void updateSceneUniformBuffer(
		const float* viewProj,
		const float* cameraPos,
		int pointLightCount,
		const float* ambient,
		int spotLightCount,
		const float* directionalDir,
		const float* directionalColor,
		float directionalIntensity);

	void updateModelUniformBuffer(const float* model, const float* albedo, float shininess, const float* specColor);
	void updateLights(const PointLight* lights, int count);
	void updateSpotLights(const SpotLight* lights, int count);

	bool createPipeline();

	void drawCube(VkCommandBuffer command_buffer);

} // namespace graphics::internal