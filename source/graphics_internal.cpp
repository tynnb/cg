#include "graphics_internal.hpp"

#include <iostream>
#include <vector>
#include <fstream>

#include <vulkan/vulkan.h>

#include <GLFW/glfw3.h>

#include <VkBootstrap.h>

#include <filesystem>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4100 4189 4324)
#endif // _MSC_VER
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif // _MSC_VER

#include <backends/imgui_impl_vulkan.h>

namespace graphics::internal {

	struct GlobalUniforms {
		float matrix[16];
		float color[4];
	};

namespace {

VkInstance vk_instance;
uint32_t vk_api_version;
VkSurfaceKHR vk_surface;

VkSwapchainKHR vk_swapchain;
std::vector<VkImage> vk_swapchain_images;
std::vector<VkImageView> vk_swapchain_image_views;
uint32_t vk_swapchain_current_image;

uint32_t vk_swapchain_resize_width;
uint32_t vk_swapchain_resize_height;
bool vk_swapchain_resize_require;

VkFormat vk_depth_buffer_format = VK_FORMAT_UNDEFINED;
VkImage vk_image_depth_buffer;
VmaAllocation vma_allocation_depth_buffer;
VkImageView vk_image_view_depth_buffer;

std::vector<VkFramebuffer> vk_framebuffers;

VkSemaphore vk_semaphore_image_available;
std::vector<VkSemaphore> vk_semaphores_image_finished;
VkFence vk_fence_frame_in_flight;

VkCommandPool vk_command_pool;
VkCommandBuffer vk_command_buffer;

VkBuffer vk_vertex_buffer;
VmaAllocation vk_vertex_buffer_allocation;

VkBuffer vk_index_buffer;
VmaAllocation vk_index_buffer_allocation;

VkBuffer vk_uniform_buffer;
VmaAllocation vk_uniform_buffer_allocation;
void* vk_uniform_buffer_mapped;

VkDescriptorSetLayout vk_descriptor_set_layout;
VkDescriptorPool vk_descriptor_pool;
VkDescriptorSet vk_descriptor_set;
VkPipelineLayout vk_pipeline_layout;
VkPipeline vk_pipeline;

VkBuffer vk_uniform_buffer_2;
VmaAllocation vk_uniform_buffer_allocation_2;
void* vk_uniform_buffer_mapped_2;

VkDescriptorSet vk_descriptor_set_2;

VkDescriptorPool vk_imgui_descriptor_pool;
VkRenderPass vk_imgui_render_pass;
std::vector<VkFramebuffer> vk_imgui_framebuffers;
VkCommandPool vk_imgui_command_pool;
VkCommandBuffer vk_imgui_command_buffer;

VkFormat selectDepthFormat(VkPhysicalDevice physical_device) {
	// Prefer the original format and preserve stencil support in the fallback.
	const VkFormat candidates[] = {
		VK_FORMAT_D24_UNORM_S8_UINT,
		VK_FORMAT_D32_SFLOAT_S8_UINT,
	};

	for (VkFormat format : candidates) {
		VkFormatProperties properties{};
		vkGetPhysicalDeviceFormatProperties(physical_device, format, &properties);
		if (properties.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
			return format;
		}
	}

	return VK_FORMAT_UNDEFINED;
}

bool initializeImGUI() {
	const VkDescriptorPoolSize descriptor_pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_SAMPLER,
			.descriptorCount = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLER_POOL_SIZE,
		},
		{
			.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
			.descriptorCount = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE,
		},
	};

	const VkDescriptorPoolCreateInfo descriptor_pool = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
		.maxSets = uint32_t(vk_swapchain_images.size()),
		.poolSizeCount = sizeof(descriptor_pool_sizes) / sizeof(descriptor_pool_sizes[0]),
		.pPoolSizes = descriptor_pool_sizes,
	};

	if (vkCreateDescriptorPool(context.device, &descriptor_pool, nullptr,
							   &vk_imgui_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool for ImGUI rendering\n";
		return false;
	}

	const VkAttachmentDescription render_pass_attachment = {
		.format = context.swapchain_format,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
		.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
		.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
		.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
		.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
	};

	const VkAttachmentReference render_pass_attachment_ref = {
		.attachment = 0,
		.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};

	const VkSubpassDescription render_pass_subpass = {
		.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
		.colorAttachmentCount = 1,
		.pColorAttachments = &render_pass_attachment_ref,
	};

	const VkSubpassDependency render_pass_dependency = {
		.srcSubpass = VK_SUBPASS_EXTERNAL,
		.dstSubpass = 0,
		.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
						 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
	};

	const VkRenderPassCreateInfo render_pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &render_pass_attachment,
		.subpassCount = 1,
		.pSubpasses = &render_pass_subpass,
		.dependencyCount = 1,
		.pDependencies = &render_pass_dependency,
	};

	if (vkCreateRenderPass(context.device, &render_pass, nullptr,
						   &vk_imgui_render_pass) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan render pass for ImGUI rendering\n";
		return false;
	}

	const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());

	vk_imgui_framebuffers.resize(swapchain_images_count);

	VkFramebufferCreateInfo framebuffer = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = vk_imgui_render_pass,
		.attachmentCount = 1,
		.width = context.swapchain_extent.width,
		.height = context.swapchain_extent.height,
		.layers = 1,
	};

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		framebuffer.pAttachments = &vk_swapchain_image_views[i];

		if (vkCreateFramebuffer(context.device, &framebuffer, nullptr,
								&vk_imgui_framebuffers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan framebuffer #" << i << " for ImGUI rendering\n";
			return false;
		}
	}

	const VkCommandPoolCreateInfo command_pool = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};

	if (vkCreateCommandPool(context.device, &command_pool, nullptr,
							&vk_imgui_command_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan command pool for ImGUI rendering\n";
		return false;
	}

	const VkCommandBufferAllocateInfo command_buffer = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk_imgui_command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	if (vkAllocateCommandBuffers(context.device, &command_buffer,
								 &vk_imgui_command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan command buffer for ImGUI rendering\n";
		return false;
	}

	ImGui_ImplVulkan_InitInfo init = {
		.ApiVersion = vk_api_version,
		.Instance = vk_instance,
		.PhysicalDevice = context.physical_device,
		.Device = context.device,
		.QueueFamily = context.graphics_queue_index,
		.Queue = context.graphics_queue,
		.DescriptorPool = vk_imgui_descriptor_pool,
		.MinImageCount = swapchain_images_count,
		.ImageCount = swapchain_images_count,
		.PipelineInfoMain = {
			.RenderPass = vk_imgui_render_pass,
		},
	};

	return ImGui_ImplVulkan_Init(&init);
}

void drawImGUI() {
	vkResetCommandBuffer(vk_imgui_command_buffer, 0);

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	vkBeginCommandBuffer(vk_imgui_command_buffer, &command_buffer_begin);

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = vk_imgui_render_pass,
		.framebuffer = vk_imgui_framebuffers[vk_swapchain_current_image],
		.renderArea = { .extent = context.swapchain_extent },
	};

	vkCmdBeginRenderPass(vk_imgui_command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), vk_imgui_command_buffer);

	vkCmdEndRenderPass(vk_imgui_command_buffer);

	vkEndCommandBuffer(vk_imgui_command_buffer);
}

bool rebuildSwapchain(uint32_t width, uint32_t height) {
	vkQueueWaitIdle(context.graphics_queue);

	vkb::SwapchainBuilder sb(context.physical_device, context.device, vk_surface,
	                         context.graphics_queue_index, context.graphics_queue_index);

	auto sb_result = sb.set_desired_extent(width, height)
	                   .use_default_format_selection()
					   .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
					   .use_default_image_usage_flags()
					   .set_old_swapchain(vk_swapchain)
					   .build();
	if (!sb_result) {
		std::cerr << sb_result.error().message() << '\n';
	}

	auto vkb_swapchain = sb_result.value();

	for (size_t i = 0, n = vk_swapchain_image_views.size(); i < n; ++i) {
		vkDestroyFramebuffer(context.device, vk_imgui_framebuffers[i], nullptr);
		vkDestroyFramebuffer(context.device, vk_framebuffers[i], nullptr);
		vkDestroyImageView(context.device, vk_swapchain_image_views[i], nullptr);
	}

	vkDestroySwapchainKHR(context.device, vk_swapchain, nullptr);

	vk_swapchain = vkb_swapchain.swapchain;
	context.swapchain_format = vkb_swapchain.image_format;
	context.swapchain_extent = vkb_swapchain.extent;

	auto swapchain_images = vkb_swapchain.get_images().value();
	auto swapchain_image_views = vkb_swapchain.get_image_views().value();

	vk_swapchain_images = std::move(swapchain_images);
	vk_swapchain_image_views = std::move(swapchain_image_views);

	vkDestroyImageView(context.device, vk_image_view_depth_buffer, nullptr);
	vmaDestroyImage(context.allocator, vk_image_depth_buffer, vma_allocation_depth_buffer);

	const VkImageCreateInfo depth_buffer = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = vk_depth_buffer_format,
		.extent = { context.swapchain_extent.width, context.swapchain_extent.height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo depth_buffer_allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateImage(context.allocator, &depth_buffer, &depth_buffer_allocation,
					   &vk_image_depth_buffer, &vma_allocation_depth_buffer,
					   nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to allocate and create Vulkan image for depth buffer\n";
		return false;
	}

	const VkImageViewCreateInfo depth_buffer_view = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = vk_image_depth_buffer,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = vk_depth_buffer_format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &depth_buffer_view, nullptr,
						  &vk_image_view_depth_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan image view for depth buffer\n";
		return false;
	}

	const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());

	VkImageView framebuffer_attachments[] = {
		VK_NULL_HANDLE,
		vk_image_view_depth_buffer,
	};

	const VkFramebufferCreateInfo framebuffer = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = context.render_pass,
		.attachmentCount = sizeof(framebuffer_attachments) / sizeof(framebuffer_attachments[0]),
		.pAttachments = framebuffer_attachments,
		.width = context.swapchain_extent.width,
		.height = context.swapchain_extent.height,
		.layers = 1,
	};

	vk_framebuffers.resize(swapchain_images_count);

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		framebuffer_attachments[0] = vk_swapchain_image_views[i];

		if (vkCreateFramebuffer(context.device, &framebuffer, nullptr,
								&vk_framebuffers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan framebuffer #" << i << '\n';
			return false;
		}
	}

	vk_imgui_framebuffers.resize(swapchain_images_count);

	VkFramebufferCreateInfo imgui_framebuffer = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = vk_imgui_render_pass,
		.attachmentCount = 1,
		.width = context.swapchain_extent.width,
		.height = context.swapchain_extent.height,
		.layers = 1,
	};

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		imgui_framebuffer.pAttachments = &vk_swapchain_image_views[i];

		if (vkCreateFramebuffer(context.device, &imgui_framebuffer, nullptr,
								&vk_imgui_framebuffers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan framebuffer #" << i << " for ImGUI rendering\n";
			return false;
		}
	}

	vk_swapchain_resize_require = false;

	return true;
}

} // namespace

Context context;

bool createBuffers(const void* vertices, size_t vertex_count, size_t vertex_size,
	const uint32_t* indices, size_t index_count) {
	const VkBufferCreateInfo vertex_buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = vertex_count * vertex_size,
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo vertex_alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo vertex_allocation_info;
	if (vmaCreateBuffer(context.allocator, &vertex_buffer_info, &vertex_alloc_info,
		&vk_vertex_buffer, &vk_vertex_buffer_allocation, &vertex_allocation_info) != VK_SUCCESS) {
		std::cerr << "Failed to create vertex buffer\n";
		return false;
	}

	memcpy(vertex_allocation_info.pMappedData, vertices, vertex_buffer_info.size);

	const VkBufferCreateInfo index_buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = index_count * sizeof(uint32_t),
		.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo index_alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo index_allocation_info;
	if (vmaCreateBuffer(context.allocator, &index_buffer_info, &index_alloc_info,
		&vk_index_buffer, &vk_index_buffer_allocation, &index_allocation_info) != VK_SUCCESS) {
		std::cerr << "Failed to create index buffer\n";
		return false;
	}

	memcpy(index_allocation_info.pMappedData, indices, index_buffer_info.size);

	return true;
}

bool createUniformBuffer() {
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(GlobalUniforms),
		.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo alloc_info = {
		.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo allocation_info;
	if (vmaCreateBuffer(context.allocator, &buffer_info, &alloc_info,
		&vk_uniform_buffer, &vk_uniform_buffer_allocation,
		&allocation_info) != VK_SUCCESS) {
		std::cerr << "Failed to create uniform buffer\n";
		return false;
	}
	vk_uniform_buffer_mapped = allocation_info.pMappedData;

	VmaAllocationInfo allocation_info_2;
	if (vmaCreateBuffer(context.allocator, &buffer_info, &alloc_info,
		&vk_uniform_buffer_2, &vk_uniform_buffer_allocation_2,
		&allocation_info_2) != VK_SUCCESS) {
		std::cerr << "Failed to create uniform buffer 2\n";
		return false;
	}
	vk_uniform_buffer_mapped_2 = allocation_info_2.pMappedData;

	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &binding,
	};

	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr,
		&vk_descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	const VkDescriptorPoolSize pool_size = {
	.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
	.descriptorCount = 2,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 2,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
		&vk_descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor pool\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo alloc_set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = vk_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &alloc_set_info,
		&vk_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set\n";
		return false;
	}

	const VkDescriptorSetAllocateInfo alloc_set_info_2 = {
	.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	.descriptorPool = vk_descriptor_pool,
	.descriptorSetCount = 1,
	.pSetLayouts = &vk_descriptor_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &alloc_set_info_2,
		&vk_descriptor_set_2) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor set 2\n";
		return false;
	}

	const VkDescriptorBufferInfo buffer_descriptor = {
		.buffer = vk_uniform_buffer,
		.offset = 0,
		.range = sizeof(GlobalUniforms),
	};

	const VkWriteDescriptorSet descriptor_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vk_descriptor_set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_descriptor,
	};

	vkUpdateDescriptorSets(context.device, 1, &descriptor_write, 0, nullptr);

	const VkDescriptorBufferInfo buffer_descriptor_2 = {
	.buffer = vk_uniform_buffer_2,
	.offset = 0,
	.range = sizeof(GlobalUniforms),
	};

	const VkWriteDescriptorSet descriptor_write_2 = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = vk_descriptor_set_2,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_descriptor_2,
	};

	vkUpdateDescriptorSets(context.device, 1, &descriptor_write_2, 0, nullptr);

	return true;
}

VkShaderModule loadShaderModule(const char* path) {
	auto fullPath = std::filesystem::absolute(path);

	std::cout << "Loading shader: " << fullPath << '\n';

	std::ifstream file(fullPath, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Failed to open shader file: " << path << '\n';
		return VK_NULL_HANDLE;
	}

	const size_t size = file.tellg();
	std::vector<char> buffer(size);
	file.seekg(0);
	file.read(buffer.data(), size);
	file.close();

	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = reinterpret_cast<const uint32_t*>(buffer.data()),
	};

	VkShaderModule module;
	if (vkCreateShaderModule(context.device, &info, nullptr, &module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module\n";
		return VK_NULL_HANDLE;
	}

	return module;
}

bool createPipeline() {
	VkShaderModule vert_shader = loadShaderModule("shaders/shader.vert.spv");
	VkShaderModule frag_shader = loadShaderModule("shaders/shader.frag.spv");
	if (vert_shader == VK_NULL_HANDLE || frag_shader == VK_NULL_HANDLE) {
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vert_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = frag_shader,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription binding = {
		.binding = 0,
		.stride = sizeof(float) * 6,
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = 0,
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = sizeof(float) * 3,
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = 2,
		.pVertexAttributeDescriptions = attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterizer = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
						  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &vk_descriptor_set_layout,
	};

	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr,
		&vk_pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";
		return false;
	}

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterizer,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = vk_pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1,
		&pipeline_info, nullptr, &vk_pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";
		return false;
	}

	vkDestroyShaderModule(context.device, vert_shader, nullptr);
	vkDestroyShaderModule(context.device, frag_shader, nullptr);

	return true;
}

void drawCube(VkCommandBuffer command_buffer) {
	vkResetCommandBuffer(command_buffer, 0);

	const VkCommandBufferBeginInfo begin_info = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	vkBeginCommandBuffer(command_buffer, &begin_info);

	const VkClearValue clear_values[] = {
		{.color = {.float32 = { 0.1f, 0.1f, 0.1f, 1.0f } } },
		{.depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = vk_framebuffers[vk_swapchain_current_image],
		.renderArea = {.extent = context.swapchain_extent },
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0, .y = 0,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0, .maxDepth = 1,
	};
	const VkRect2D scissor = { .extent = context.swapchain_extent };
	vkCmdSetViewport(command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(command_buffer, 0, 1, &scissor);

	vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(command_buffer, 0, 1, &vk_vertex_buffer, &offset);
	vkCmdBindIndexBuffer(command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

	vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		vk_pipeline_layout, 0, 1, &vk_descriptor_set, 0, nullptr);
	vkCmdDrawIndexed(command_buffer, 36, 1, 0, 0, 0);

	vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
		vk_pipeline_layout, 0, 1, &vk_descriptor_set_2, 0, nullptr);
	vkCmdDrawIndexed(command_buffer, 36, 1, 0, 0, 0);

	vkCmdEndRenderPass(command_buffer);
	vkEndCommandBuffer(command_buffer);
}

void updateUniformBuffer(const float* matrix, const float* color) {
	char* dst = (char*)vk_uniform_buffer_mapped;
	memcpy(dst, matrix, 64);
	memcpy(dst + 64, color, 16);
}

void updateUniformBuffer2(const float* matrix, const float* color) {
	char* dst = (char*)vk_uniform_buffer_mapped_2;
	memcpy(dst, matrix, 64);
	memcpy(dst + 64, color, 16);
}

bool initialize(GLFWwindow* const window) {
	vkb::InstanceBuilder ib;

	auto ibr = ib.require_api_version(VK_MAKE_VERSION(1, 1, 0))
				 .request_validation_layers()
				 .build();

	auto vkb_instance = ibr.value();
	vk_instance = vkb_instance.instance;
	vk_api_version = vkb_instance.api_version;

	if (glfwCreateWindowSurface(vk_instance, window, nullptr, &vk_surface) != VK_SUCCESS) {
		const char *message = nullptr;
		glfwGetError(&message);
		std::cerr << message << '\n';
		return false;
	}

	vkb::PhysicalDeviceSelector pds(vkb_instance, vk_surface);

	auto pds_result = pds.prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
						 .require_present()
						 .select();
	if (!pds_result) {
		std::cerr << pds_result.error().message() << '\n';
		return false;
	}

	auto vkb_physical_device = pds_result.value();

	vk_depth_buffer_format = selectDepthFormat(vkb_physical_device.physical_device);
	if (vk_depth_buffer_format == VK_FORMAT_UNDEFINED) {
		std::cerr << "No supported depth/stencil attachment format found\n";
		return false;
	}

	vkb::DeviceBuilder db(vkb_physical_device);

	auto db_result = db.build();
	if (!db_result) {
		std::cerr << db_result.error().message() << '\n';
		return false;
	}

	auto vkb_device = db_result.value();

	context.physical_device = vkb_device.physical_device;
	context.device = vkb_device.device;

	if (auto result = vkb_device.get_queue(vkb::QueueType::graphics); result) {
		context.graphics_queue = result.value();
	} else {
		std::cerr << result.error().message() << '\n';
		return false;
	}

	if (auto result = vkb_device.get_queue_index(vkb::QueueType::graphics); result) {
		context.graphics_queue_index = result.value();
	} else {
		std::cerr << result.error().message() << '\n';
		return false;
	}

	const VmaAllocatorCreateInfo allocator = {
		.physicalDevice = context.physical_device,
		.device = context.device,
		.instance = vk_instance,
		.vulkanApiVersion = vk_api_version,
	};

	if (vmaCreateAllocator(&allocator, &context.allocator) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan Memory Allocator\n";
		return false;
	}

	vkb::SwapchainBuilder sb(vkb_device);

	auto sb_result = sb.use_default_format_selection()
					   .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
					   .use_default_image_usage_flags()
					   .build();
	if (!sb_result) {
		std::cerr << sb_result.error().message() << '\n';
		return false;
	}

	auto vkb_swapchain = sb_result.value();

	vk_swapchain = vkb_swapchain.swapchain;
	context.swapchain_format = vkb_swapchain.image_format;
	context.swapchain_extent = vkb_swapchain.extent;
	vk_swapchain_images = vkb_swapchain.get_images().value();
	vk_swapchain_image_views = vkb_swapchain.get_image_views().value();
	vk_swapchain_current_image = UINT32_MAX;

	const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());

	const VkImageCreateInfo depth_buffer = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D,
		.format = vk_depth_buffer_format,
		.extent = { context.swapchain_extent.width, context.swapchain_extent.height, 1 },
		.mipLevels = 1,
		.arrayLayers = 1,
		.samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};

	const VmaAllocationCreateInfo depth_buffer_allocation = {
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	if (vmaCreateImage(context.allocator, &depth_buffer, &depth_buffer_allocation,
					   &vk_image_depth_buffer, &vma_allocation_depth_buffer,
					   nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to allocate and create Vulkan image for depth buffer\n";
		return false;
	}

	const VkImageViewCreateInfo depth_buffer_view = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = vk_image_depth_buffer,
		.viewType = VK_IMAGE_VIEW_TYPE_2D,
		.format = vk_depth_buffer_format,
		.subresourceRange = {
			.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
			.baseMipLevel = 0,
			.levelCount = 1,
			.baseArrayLayer = 0,
			.layerCount = 1,
		},
	};

	if (vkCreateImageView(context.device, &depth_buffer_view, nullptr,
						  &vk_image_view_depth_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan image view for depth buffer\n";
		return false;
	}

	const VkAttachmentDescription render_pass_attachments[] = {
		{
			.format = context.swapchain_format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		},
		{
			.format = vk_depth_buffer_format,
			.samples = VK_SAMPLE_COUNT_1_BIT,
			.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
			.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
			.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
			.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
			.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		},
	};

	const VkAttachmentReference render_pass_color_attachment = {
		.attachment = 0,
		.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	};

	const VkAttachmentReference render_pass_depth_attachment = {
		.attachment = 1,
		.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	};

	const VkSubpassDescription render_pass_subpass = {
		.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
		.colorAttachmentCount = 1,
		.pColorAttachments = &render_pass_color_attachment,
		.pDepthStencilAttachment = &render_pass_depth_attachment,
	};

	const VkRenderPassCreateInfo render_pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
		.attachmentCount = sizeof(render_pass_attachments) / sizeof(render_pass_attachments[0]),
		.pAttachments = render_pass_attachments,
		.subpassCount = 1,
		.pSubpasses = &render_pass_subpass,
	};

	if (vkCreateRenderPass(context.device, &render_pass, nullptr, &context.render_pass) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan render pass\n";
		return false;
	}

	VkImageView framebuffer_attachments[] = {
		VK_NULL_HANDLE,
		vk_image_view_depth_buffer
	};

	const VkFramebufferCreateInfo framebuffer = {
		.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		.renderPass = context.render_pass,
		.attachmentCount = sizeof(framebuffer_attachments) / sizeof(framebuffer_attachments[0]),
		.pAttachments = framebuffer_attachments,
		.width = context.swapchain_extent.width,
		.height = context.swapchain_extent.height,
		.layers = 1,
	};

	vk_framebuffers.resize(swapchain_images_count);

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		framebuffer_attachments[0] = vk_swapchain_image_views[i];

		if (vkCreateFramebuffer(context.device, &framebuffer, nullptr,
								&vk_framebuffers[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan framebuffer #" << i << '\n';
			return false;
		}
	}

	const VkSemaphoreCreateInfo semaphore = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };

	const VkFenceCreateInfo fence = {
		.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
		.flags = VK_FENCE_CREATE_SIGNALED_BIT,
	};

	vk_semaphores_image_finished.resize(swapchain_images_count);

	for (uint32_t i = 0; i < swapchain_images_count; ++i) {
		if (vkCreateSemaphore(context.device, &semaphore, nullptr,
							  &vk_semaphores_image_finished[i]) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan semaphore #" << i << " for finished image\n";
			return false;
		}
	}

	if (vkCreateSemaphore(context.device, &semaphore, NULL,
						  &vk_semaphore_image_available) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan semaphore for available image\n";
		return false;
	}

	if (vkCreateFence(context.device, &fence, nullptr, &vk_fence_frame_in_flight) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan semaphore for image in flight\n";
		return false;
	}

	const VkCommandPoolCreateInfo command_pool = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
		.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
		.queueFamilyIndex = context.graphics_queue_index,
	};

	if (vkCreateCommandPool(context.device, &command_pool, nullptr, &vk_command_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan command pool\n";
		return false;
	}

	const VkCommandBufferAllocateInfo command_buffers = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		.commandPool = vk_command_pool,
		.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		.commandBufferCount = 1,
	};

	if (vkAllocateCommandBuffers(context.device, &command_buffers, &vk_command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan command buffer\n";
		return false;
	}

	if (!initializeImGUI()) {
		std::cerr << "Failed to initialize ImGUI Vulkan rendering backend\n";
		return false;
	}

	return true;
}

void shutdown() {
	vkQueueWaitIdle(context.graphics_queue);

	ImGui_ImplVulkan_Shutdown();

	vkDestroyCommandPool(context.device, vk_imgui_command_pool, nullptr);
	for (size_t i = 0, n = vk_imgui_framebuffers.size(); i < n; ++i) {
		vkDestroyFramebuffer(context.device, vk_imgui_framebuffers[i], nullptr);
	}
	vkDestroyRenderPass(context.device, vk_imgui_render_pass, nullptr);
	vkDestroyDescriptorPool(context.device, vk_imgui_descriptor_pool, nullptr);

	vkDestroyCommandPool(context.device, vk_command_pool, nullptr);

	vkDestroyFence(context.device, vk_fence_frame_in_flight, nullptr);
	for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
		vkDestroySemaphore(context.device, vk_semaphores_image_finished[i], nullptr);
	}
	vkDestroySemaphore(context.device, vk_semaphore_image_available, nullptr);

	for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
		vkDestroyFramebuffer(context.device, vk_framebuffers[i], nullptr);
	}
	vkDestroyRenderPass(context.device, context.render_pass, nullptr);

	vkDestroyImageView(context.device, vk_image_view_depth_buffer, nullptr);
	vmaDestroyImage(context.allocator, vk_image_depth_buffer, vma_allocation_depth_buffer);

	for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) {
		vkDestroyImageView(context.device, vk_swapchain_image_views[i], nullptr);
	}
	vkDestroySwapchainKHR(context.device, vk_swapchain, nullptr);

	vkDestroyPipeline(context.device, vk_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);

	vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout, nullptr);

	vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
	vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation);
	vmaDestroyBuffer(context.allocator, vk_uniform_buffer, vk_uniform_buffer_allocation);
	vmaDestroyBuffer(context.allocator, vk_uniform_buffer_2, vk_uniform_buffer_allocation_2);

	vmaDestroyAllocator(context.allocator);

	vkDestroyDevice(context.device, nullptr);

	vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
	vkDestroyInstance(vk_instance, nullptr);
}

void resize(uint32_t width, uint32_t height) {
	if (width == 0 || height == 0) {
		return;
	}

	vk_swapchain_resize_width = width;
	vk_swapchain_resize_height = height;

	vk_swapchain_resize_require = true;
}

FrameData prepare() {
	vkWaitForFences(context.device, 1, &vk_fence_frame_in_flight, VK_TRUE, UINT64_MAX);

retry_acquire:
	switch (vkAcquireNextImageKHR(context.device, vk_swapchain, UINT64_MAX,
								  vk_semaphore_image_available, VK_NULL_HANDLE,
								  &vk_swapchain_current_image)) {
	case VK_SUCCESS:
		break;

	case VK_ERROR_OUT_OF_DATE_KHR:
		rebuildSwapchain(vk_swapchain_resize_width, vk_swapchain_resize_height);
		goto retry_acquire;

	case VK_SUBOPTIMAL_KHR:
		std::cerr << "Swapchain is suboptimal for rendering!\n";
		break;

	default:
		std::cerr << "Failed to present Vulkan swapchain image\n";
		return {};
	}

	vkResetFences(context.device, 1, &vk_fence_frame_in_flight);

	drawCube(vk_command_buffer);

	return {
		.framebuffer = vk_framebuffers[vk_swapchain_current_image],
		.command_buffer = vk_command_buffer,
	};
}

void submitAndPresent() {
	drawImGUI();

	const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	const VkCommandBuffer command_buffers[] = {
		vk_command_buffer,
		vk_imgui_command_buffer,
	};

	const VkSubmitInfo submit = {
		.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &vk_semaphore_image_available,
		.pWaitDstStageMask = &stage,
		.commandBufferCount = sizeof(command_buffers) / sizeof(command_buffers[0]),
		.pCommandBuffers = command_buffers,
		.signalSemaphoreCount = 1,
		.pSignalSemaphores = &vk_semaphores_image_finished[vk_swapchain_current_image],
	};

	vkQueueSubmit(context.graphics_queue, 1, &submit, vk_fence_frame_in_flight);

	const VkPresentInfoKHR present = {
		.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
		.waitSemaphoreCount = 1,
		.pWaitSemaphores = &vk_semaphores_image_finished[vk_swapchain_current_image],
		.swapchainCount = 1,
		.pSwapchains = &vk_swapchain,
		.pImageIndices = &vk_swapchain_current_image,
	};

	VkResult result = vkQueuePresentKHR(context.graphics_queue, &present);
	if (result == VK_ERROR_OUT_OF_DATE_KHR ||
	    result == VK_SUBOPTIMAL_KHR ||
	    vk_swapchain_resize_require) {
		rebuildSwapchain(vk_swapchain_resize_width, vk_swapchain_resize_height);
	} else {
		std::cerr << "Failed to present Vulkan swapchain image\n";
	}
}

} // namespace graphics::internal
