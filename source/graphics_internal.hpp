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

	extern Context context;

	bool initialize(GLFWwindow* const window);
	void shutdown();

	void resize(uint32_t width, uint32_t height);

	FrameData prepare();
	void submitAndPresent();

	bool createBuffers(const void* vertices, size_t vertex_count, size_t vertex_size,
		const uint32_t* indices, size_t index_count);

	bool createUniformBuffer();
	void updateUniformBuffer(const float* matrix, const float* color);
	void updateUniformBuffer2(const float* matrix, const float* color);

	bool createPipeline();

	void drawCube(VkCommandBuffer command_buffer);

} // namespace graphics::internal