#include "graphics_internal.hpp"

#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <algorithm>

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

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

namespace graphics::internal {

	struct SceneUniforms {
		float viewProj[16];
		float cameraPos[4];
		float ambientColor[4];
		float directionalDirection[4];
		float directionalColor[4];
		float lightViewProj[16];
		int pointLightCount;
		int spotLightCount;
		float _pad[2];
	};

	struct ModelUniforms {
		float model[16];
		float albedo[3];
		float shininess;
		float specularColor[3];
		int useTexture;
	};

	static_assert(sizeof(SceneUniforms) % 16 == 0, "SceneUniforms must be 16-byte aligned");
	static_assert(sizeof(ModelUniforms) % 16 == 0, "ModelUniforms must be 16-byte aligned");

	constexpr uint32_t MAX_POINT_LIGHTS = 16;
	constexpr uint32_t MAX_SPOT_LIGHTS = 16;

	namespace {
        constexpr uint32_t MAX_TEXTURES = 8;

        struct Texture {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkSampler sampler = VK_NULL_HANDLE;
            VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
        };

        std::vector<Texture> g_textures;
        int g_activeTextureIndex = 0;

        VkDescriptorSetLayout vk_descriptor_set_layout_texture;

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

		constexpr uint32_t SHADOW_MAP_SIZE = 2048;
		const VkFormat SHADOW_MAP_FORMAT = VK_FORMAT_D16_UNORM;

		VkImage vk_image_shadow_map = VK_NULL_HANDLE;
		VmaAllocation vma_allocation_shadow_map = VK_NULL_HANDLE;
		VkImageView vk_image_view_shadow_map = VK_NULL_HANDLE;
		VkSampler vk_sampler_shadow = VK_NULL_HANDLE;

		VkPipeline vk_pipeline_shadow = VK_NULL_HANDLE;
		VkPipelineLayout vk_pipeline_layout_shadow = VK_NULL_HANDLE;

		float g_lightViewProj[16] = {};
		float g_depthBiasConstant = 1.25f;
		float g_depthBiasSlope = 1.75f;
		float g_pushShadowBias = 0.001f;
		int g_drawModelCount = 1;
		size_t g_model_stride = 0;

		VkBuffer vk_vertex_buffer;
		VmaAllocation vk_vertex_buffer_allocation;

		VkBuffer vk_index_buffer;
		VmaAllocation vk_index_buffer_allocation;

		VkBuffer vk_uniform_buffer_scene;
		VmaAllocation vk_uniform_buffer_scene_allocation;
		void* vk_uniform_buffer_scene_mapped;

		VkBuffer vk_storage_buffer_lights;
		VmaAllocation vk_storage_buffer_lights_allocation;
		void* vk_storage_buffer_lights_mapped;

		VkBuffer vk_storage_buffer_spot_lights;
		VmaAllocation vk_storage_buffer_spot_lights_allocation;
		void* vk_storage_buffer_spot_lights_mapped;

		VkDescriptorSetLayout vk_descriptor_set_layout_scene;
		VkDescriptorSet vk_descriptor_set_scene;

		VkBuffer vk_uniform_buffer_model;
		VmaAllocation vk_uniform_buffer_model_allocation;
		void* vk_uniform_buffer_model_mapped;

		VkDescriptorSetLayout vk_descriptor_set_layout_model;
		VkDescriptorSet vk_descriptor_set_model;

		VkDescriptorPool vk_descriptor_pool;
		VkPipelineLayout vk_pipeline_layout;
		VkPipeline vk_pipeline;

		VkDescriptorPool vk_imgui_descriptor_pool;
		VkRenderPass vk_imgui_render_pass;
		std::vector<VkFramebuffer> vk_imgui_framebuffers;
		VkCommandPool vk_imgui_command_pool;
		VkCommandBuffer vk_imgui_command_buffer;

		VkFormat selectDepthFormat(VkPhysicalDevice physical_device) {
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

	size_t alignedModelSize() {
		size_t align = std::max<size_t>(1, context.min_uniform_buffer_offset_alignment);
		return (sizeof(ModelUniforms) + align - 1) / align * align;
	}

	bool createShadowResources() {
		VkImageCreateInfo img{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
		img.imageType = VK_IMAGE_TYPE_2D;
		img.format = SHADOW_MAP_FORMAT;
		img.extent = { SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1 };
		img.mipLevels = 1;
		img.arrayLayers = 1;
		img.samples = VK_SAMPLE_COUNT_1_BIT;
		img.tiling = VK_IMAGE_TILING_OPTIMAL;
		img.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo ai{};
		ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
		if (vmaCreateImage(context.allocator, &img, &ai,
						&vk_image_shadow_map, &vma_allocation_shadow_map, nullptr) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow map image\n"; return false;
		}

		VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
		view.image = vk_image_shadow_map;
		view.viewType = VK_IMAGE_VIEW_TYPE_2D;
		view.format = SHADOW_MAP_FORMAT;
		view.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
		if (vkCreateImageView(context.device, &view, nullptr, &vk_image_view_shadow_map) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow map image view\n"; return false;
		}

		VkSamplerCreateInfo s{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
		s.magFilter = VK_FILTER_LINEAR;
		s.minFilter = VK_FILTER_LINEAR;
		s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
		s.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		s.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		s.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
		s.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		s.anisotropyEnable = VK_FALSE;
		s.compareEnable = VK_TRUE;
		s.compareOp = VK_COMPARE_OP_LESS;
		if (vkCreateSampler(context.device, &s, nullptr, &vk_sampler_shadow) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow sampler\n"; return false;
		}
		return true;
	}

		bool initializeImGUI() {
			const VkDescriptorPoolSize descriptor_pool_sizes[] = {
				{.type = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLER_POOL_SIZE, },
				{.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = IMGUI_IMPL_VULKAN_MINIMUM_SAMPLED_IMAGE_POOL_SIZE, },
			};

			const VkDescriptorPoolCreateInfo descriptor_pool = {
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
				.maxSets = uint32_t(vk_swapchain_images.size()),
				.poolSizeCount = sizeof(descriptor_pool_sizes) / sizeof(descriptor_pool_sizes[0]),
				.pPoolSizes = descriptor_pool_sizes,
			};

			if (vkCreateDescriptorPool(context.device, &descriptor_pool, nullptr, &vk_imgui_descriptor_pool) != VK_SUCCESS) {
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
				.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
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

			if (vkCreateRenderPass(context.device, &render_pass, nullptr, &vk_imgui_render_pass) != VK_SUCCESS) {
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
				if (vkCreateFramebuffer(context.device, &framebuffer, nullptr, &vk_imgui_framebuffers[i]) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan framebuffer #" << i << " for ImGUI rendering\n";
					return false;
				}
			}

			const VkCommandPoolCreateInfo command_pool = {
				.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
				.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
				.queueFamilyIndex = context.graphics_queue_index,
			};

			if (vkCreateCommandPool(context.device, &command_pool, nullptr, &vk_imgui_command_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan command pool for ImGUI rendering\n";
				return false;
			}

			const VkCommandBufferAllocateInfo command_buffer = {
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
				.commandPool = vk_imgui_command_pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1,
			};

			if (vkAllocateCommandBuffers(context.device, &command_buffer, &vk_imgui_command_buffer) != VK_SUCCESS) {
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
				.PipelineInfoMain = {.RenderPass = vk_imgui_render_pass, },
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
				.renderArea = {.extent = context.swapchain_extent },
			};
			vkCmdBeginRenderPass(vk_imgui_command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);
			ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), vk_imgui_command_buffer);
			vkCmdEndRenderPass(vk_imgui_command_buffer);
			vkEndCommandBuffer(vk_imgui_command_buffer);
		}

		bool rebuildSwapchain(uint32_t width, uint32_t height) {
			vkQueueWaitIdle(context.graphics_queue);
			vkb::SwapchainBuilder sb(context.physical_device, context.device, vk_surface, context.graphics_queue_index, context.graphics_queue_index);
			auto sb_result = sb.set_desired_extent(width, height).use_default_format_selection().set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR).use_default_image_usage_flags().set_old_swapchain(vk_swapchain).build();
			if (!sb_result) { std::cerr << sb_result.error().message() << '\n'; }
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
				.mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
				.tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
				.sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
			};
			const VmaAllocationCreateInfo depth_buffer_allocation = { .usage = VMA_MEMORY_USAGE_AUTO, };
			if (vmaCreateImage(context.allocator, &depth_buffer, &depth_buffer_allocation, &vk_image_depth_buffer, &vma_allocation_depth_buffer, nullptr) != VK_SUCCESS) {
				std::cerr << "Failed to allocate and create Vulkan image for depth buffer\n";
				return false;
			}
			const VkImageViewCreateInfo depth_buffer_view = {
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.image = vk_image_depth_buffer, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = vk_depth_buffer_format,
				.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1, },
			};
			if (vkCreateImageView(context.device, &depth_buffer_view, nullptr, &vk_image_view_depth_buffer) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan image view for depth buffer\n";
				return false;
			}
			const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());
			VkImageView framebuffer_attachments[] = { VK_NULL_HANDLE, vk_image_view_depth_buffer, };
			const VkFramebufferCreateInfo framebuffer = {
				.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = context.render_pass,
				.attachmentCount = sizeof(framebuffer_attachments) / sizeof(framebuffer_attachments[0]),
				.pAttachments = framebuffer_attachments, .width = context.swapchain_extent.width,
				.height = context.swapchain_extent.height, .layers = 1,
			};
			vk_framebuffers.resize(swapchain_images_count);
			for (uint32_t i = 0; i < swapchain_images_count; ++i) {
				framebuffer_attachments[0] = vk_swapchain_image_views[i];
				if (vkCreateFramebuffer(context.device, &framebuffer, nullptr, &vk_framebuffers[i]) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan framebuffer #" << i << '\n';
					return false;
				}
			}
			vk_imgui_framebuffers.resize(swapchain_images_count);
			VkFramebufferCreateInfo imgui_framebuffer = {
				.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = vk_imgui_render_pass,
				.attachmentCount = 1, .width = context.swapchain_extent.width,
				.height = context.swapchain_extent.height, .layers = 1,
			};
			for (uint32_t i = 0; i < swapchain_images_count; ++i) {
				imgui_framebuffer.pAttachments = &vk_swapchain_image_views[i];
				if (vkCreateFramebuffer(context.device, &imgui_framebuffer, nullptr, &vk_imgui_framebuffers[i]) != VK_SUCCESS) {
					std::cerr << "Failed to create Vulkan framebuffer #" << i << " for ImGUI rendering\n";
					return false;
				}
			}
			vk_swapchain_resize_require = false;
			return true;
		}

		bool copyBuffer(
            VkBuffer sourceBuffer,
            VkBuffer destinationBuffer,
            VkDeviceSize size
        ) {
            VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = vk_command_pool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;

            if (vkAllocateCommandBuffers(
                context.device,
                &allocInfo,
                &commandBuffer
            ) != VK_SUCCESS) {
                std::cerr << "Failed to allocate copy command buffer\n";
                return false;
            }

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
                vkFreeCommandBuffers(
                    context.device,
                    vk_command_pool,
                    1,
                    &commandBuffer
                );
                std::cerr << "Failed to begin copy command buffer\n";
                return false;
            }

            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = size;

            vkCmdCopyBuffer(
                commandBuffer,
                sourceBuffer,
                destinationBuffer,
                1,
                &copyRegion
            );

            if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
                vkFreeCommandBuffers(
                    context.device,
                    vk_command_pool,
                    1,
                    &commandBuffer
                );
                std::cerr << "Failed to end copy command buffer\n";
                return false;
            }

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &commandBuffer;

            VkResult submitResult = vkQueueSubmit(
                context.graphics_queue,
                1,
                &submitInfo,
                VK_NULL_HANDLE
            );

            if (submitResult != VK_SUCCESS) {
                vkFreeCommandBuffers(
                    context.device,
                    vk_command_pool,
                    1,
                    &commandBuffer
                );
                std::cerr << "Failed to submit buffer copy command\n";
                return false;
            }

            VkResult waitResult = vkQueueWaitIdle(context.graphics_queue);

            vkFreeCommandBuffers(
                context.device,
                vk_command_pool,
                1,
                &commandBuffer
            );

            if (waitResult != VK_SUCCESS) {
                std::cerr << "Failed to wait for buffer copy\n";
                return false;
            }

            return true;
        }

        bool createImageFromPixels(
            const unsigned char* pixels,
            int width, int height,
            Texture& outTexture)
        {
            const VkDeviceSize imageSize = VkDeviceSize(width) * height * 4;

            VkBuffer stagingBuffer = VK_NULL_HANDLE;
            VmaAllocation stagingAllocation = VK_NULL_HANDLE;
            VkBufferCreateInfo stagingInfo{};
            stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            stagingInfo.size = imageSize;
            stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo stagingAlloc{};
            stagingAlloc.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                                 VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
            stagingAlloc.usage = VMA_MEMORY_USAGE_AUTO;

            VmaAllocationInfo stagingAllocInfo{};
            if (vmaCreateBuffer(context.allocator, &stagingInfo, &stagingAlloc,
                                &stagingBuffer, &stagingAllocation, &stagingAllocInfo) != VK_SUCCESS) {
                std::cerr << "Failed to create texture staging buffer\n";
                return false;
            }
            memcpy(stagingAllocInfo.pMappedData, pixels, static_cast<size_t>(imageSize));
            vmaFlushAllocation(context.allocator, stagingAllocation, 0, imageSize);

            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
            imageInfo.extent = { uint32_t(width), uint32_t(height), 1 };
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            VmaAllocationCreateInfo imageAlloc{};
            imageAlloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

            if (vmaCreateImage(context.allocator, &imageInfo, &imageAlloc,
                               &outTexture.image, &outTexture.allocation, nullptr) != VK_SUCCESS) {
                vmaDestroyBuffer(context.allocator, stagingBuffer, stagingAllocation);
                std::cerr << "Failed to create texture image\n";
                return false;
            }

            VkCommandBuffer cmd = VK_NULL_HANDLE;
            VkCommandBufferAllocateInfo cmdAlloc{};
            cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmdAlloc.commandPool = vk_command_pool;
            cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAlloc.commandBufferCount = 1;
            vkAllocateCommandBuffers(context.device, &cmdAlloc, &cmd);

            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &begin);

            VkImageMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = outTexture.image;
            barrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);

            VkBufferImageCopy region{};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.imageOffset = { 0, 0, 0 };
            region.imageExtent = { uint32_t(width), uint32_t(height), 1 };

            vkCmdCopyBufferToImage(cmd, stagingBuffer, outTexture.image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            VkImageMemoryBarrier barrier2 = barrier;
            barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier2);

            vkEndCommandBuffer(cmd);

            VkSubmitInfo submit{};
            submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &cmd;
            vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE);
            vkQueueWaitIdle(context.graphics_queue);
            vkFreeCommandBuffers(context.device, vk_command_pool, 1, &cmd);

            vmaDestroyBuffer(context.allocator, stagingBuffer, stagingAllocation);

            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = outTexture.image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
            viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            if (vkCreateImageView(context.device, &viewInfo, nullptr, &outTexture.view) != VK_SUCCESS) {
                std::cerr << "Failed to create texture image view\n";
                return false;
            }

            VkSamplerCreateInfo samplerInfo{};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.anisotropyEnable = VK_FALSE;
            samplerInfo.maxAnisotropy = 1.0f;
            samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            samplerInfo.unnormalizedCoordinates = VK_FALSE;
            samplerInfo.compareEnable = VK_FALSE;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

            if (vkCreateSampler(context.device, &samplerInfo, nullptr, &outTexture.sampler) != VK_SUCCESS) {
                std::cerr << "Failed to create texture sampler\n";
                return false;
            }

            VkDescriptorSetAllocateInfo setAlloc{};
            setAlloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            setAlloc.descriptorPool = vk_descriptor_pool;
            setAlloc.descriptorSetCount = 1;
            setAlloc.pSetLayouts = &vk_descriptor_set_layout_texture;

            if (vkAllocateDescriptorSets(context.device, &setAlloc, &outTexture.descriptorSet) != VK_SUCCESS) {
                std::cerr << "Failed to allocate texture descriptor set\n";
                return false;
            }

            VkDescriptorImageInfo imageDescriptor{};
            imageDescriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageDescriptor.imageView = outTexture.view;
            imageDescriptor.sampler = outTexture.sampler;

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = outTexture.descriptorSet;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &imageDescriptor;

            vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);

            return true;
        }

	} // namespace

	bool createTextureFromFile(const char* path) {
        int width = 0, height = 0, channels = 0;
        stbi_set_flip_vertically_on_load(true);
        unsigned char* pixels = stbi_load(path, &width, &height, &channels, 4);
        if (!pixels) {
            std::cerr << "Failed to load texture: " << path << '\n';
            return false;
        }

        Texture texture;
        bool ok = createImageFromPixels(pixels, width, height, texture);
        stbi_image_free(pixels);

        if (ok) {
            g_textures.push_back(texture);
        }
        return ok;
    }

    void setActiveTexture(int index) {
        if (index >= 0 && index < (int)g_textures.size()) {
            g_activeTextureIndex = index;
        }
    }

	namespace {
        float g_pushTime = 0.0f;
        int g_pushEnableWeirdSampling = 0;
    }

    void setPushConstants(float time, int enableWeirdSampling) {
        g_pushTime = time;
        g_pushEnableWeirdSampling = enableWeirdSampling;
    }

	void setShadowParams(const float* lightViewProj,
						float depthBiasConstant,
						float depthBiasSlope,
						float shadowBias) {
		memcpy(g_lightViewProj, lightViewProj, sizeof(g_lightViewProj));
		g_depthBiasConstant = depthBiasConstant;
		g_depthBiasSlope    = depthBiasSlope;
		g_pushShadowBias    = shadowBias;
	}

	void setDrawModelCount(int count) {
		g_drawModelCount = std::max(1, count);
	}

    int getActiveTextureIndex() {
        return g_activeTextureIndex;
    }

    int getTextureCount() {
        return (int)g_textures.size();
    }

	Context context;

	bool createBuffers(
        const void* vertices,
        size_t vertex_count,
        size_t vertex_size,
        const uint32_t* indices,
        size_t index_count
    ) {
        VkDeviceSize vertexBufferSize = vertex_count * vertex_size;
        VkDeviceSize indexBufferSize = index_count * sizeof(uint32_t);

        VkBuffer stagingVertexBuffer = VK_NULL_HANDLE;
        VmaAllocation stagingVertexAllocation = VK_NULL_HANDLE;

        VkBufferCreateInfo stagingBufferInfo{};
        stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingBufferInfo.size = vertexBufferSize;
        stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo stagingAllocationInfo{};
        stagingAllocationInfo.flags =
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        stagingAllocationInfo.usage = VMA_MEMORY_USAGE_AUTO;

        VmaAllocationInfo vertexAllocationInfo{};

        if (vmaCreateBuffer(
            context.allocator,
            &stagingBufferInfo,
            &stagingAllocationInfo,
            &stagingVertexBuffer,
            &stagingVertexAllocation,
            &vertexAllocationInfo
        ) != VK_SUCCESS) {
            std::cerr << "Failed to create vertex staging buffer\n";
            return false;
        }

        memcpy(
            vertexAllocationInfo.pMappedData,
            vertices,
            static_cast<size_t>(vertexBufferSize)
        );

        vmaFlushAllocation(
            context.allocator,
            stagingVertexAllocation,
            0,
            vertexBufferSize
        );

        VkBufferCreateInfo vertexBufferInfo{};
        vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        vertexBufferInfo.size = vertexBufferSize;
        vertexBufferInfo.usage =
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        vertexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo gpuAllocationInfo{};
        gpuAllocationInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        if (vmaCreateBuffer(
            context.allocator,
            &vertexBufferInfo,
            &gpuAllocationInfo,
            &vk_vertex_buffer,
            &vk_vertex_buffer_allocation,
            nullptr
        ) != VK_SUCCESS) {
            std::cerr << "Failed to create GPU vertex buffer\n";

            vmaDestroyBuffer(
                context.allocator,
                stagingVertexBuffer,
                stagingVertexAllocation
            );

            return false;
        }

        if (!copyBuffer(
            stagingVertexBuffer,
            vk_vertex_buffer,
            vertexBufferSize
        )) {
            vmaDestroyBuffer(
                context.allocator,
                stagingVertexBuffer,
                stagingVertexAllocation
            );

            vmaDestroyBuffer(
                context.allocator,
                vk_vertex_buffer,
                vk_vertex_buffer_allocation
            );

            vk_vertex_buffer = VK_NULL_HANDLE;
            vk_vertex_buffer_allocation = VK_NULL_HANDLE;

            return false;
        }

        vmaDestroyBuffer(
            context.allocator,
            stagingVertexBuffer,
            stagingVertexAllocation
        );

        VkBuffer stagingIndexBuffer = VK_NULL_HANDLE;
        VmaAllocation stagingIndexAllocation = VK_NULL_HANDLE;

        stagingBufferInfo.size = indexBufferSize;

        if (vmaCreateBuffer(
            context.allocator,
            &stagingBufferInfo,
            &stagingAllocationInfo,
            &stagingIndexBuffer,
            &stagingIndexAllocation,
            &vertexAllocationInfo
        ) != VK_SUCCESS) {
            std::cerr << "Failed to create index staging buffer\n";

            vmaDestroyBuffer(
                context.allocator,
                vk_vertex_buffer,
                vk_vertex_buffer_allocation
            );

            vk_vertex_buffer = VK_NULL_HANDLE;
            vk_vertex_buffer_allocation = VK_NULL_HANDLE;

            return false;
        }

        memcpy(
            vertexAllocationInfo.pMappedData,
            indices,
            static_cast<size_t>(indexBufferSize)
        );

        vmaFlushAllocation(
            context.allocator,
            stagingIndexAllocation,
            0,
            indexBufferSize
        );

        VkBufferCreateInfo indexBufferInfo{};
        indexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        indexBufferInfo.size = indexBufferSize;
        indexBufferInfo.usage =
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        indexBufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vmaCreateBuffer(
            context.allocator,
            &indexBufferInfo,
            &gpuAllocationInfo,
            &vk_index_buffer,
            &vk_index_buffer_allocation,
            nullptr
        ) != VK_SUCCESS) {
            std::cerr << "Failed to create GPU index buffer\n";

            vmaDestroyBuffer(
                context.allocator,
                stagingIndexBuffer,
                stagingIndexAllocation
            );

            vmaDestroyBuffer(
                context.allocator,
                vk_vertex_buffer,
                vk_vertex_buffer_allocation
            );

            vk_vertex_buffer = VK_NULL_HANDLE;
            vk_vertex_buffer_allocation = VK_NULL_HANDLE;

            return false;
        }

        if (!copyBuffer(
            stagingIndexBuffer,
            vk_index_buffer,
            indexBufferSize
        )) {
            vmaDestroyBuffer(
                context.allocator,
                stagingIndexBuffer,
                stagingIndexAllocation
            );

            vmaDestroyBuffer(
                context.allocator,
                vk_index_buffer,
                vk_index_buffer_allocation
            );

            vmaDestroyBuffer(
                context.allocator,
                vk_vertex_buffer,
                vk_vertex_buffer_allocation
            );

            vk_index_buffer = VK_NULL_HANDLE;
            vk_index_buffer_allocation = VK_NULL_HANDLE;
            vk_vertex_buffer = VK_NULL_HANDLE;
            vk_vertex_buffer_allocation = VK_NULL_HANDLE;

            return false;
        }

        vmaDestroyBuffer(
            context.allocator,
            stagingIndexBuffer,
            stagingIndexAllocation
        );

        return true;
    }

	bool createUniformBuffer() {
		if (!createShadowResources()) { return false; }
		VkBufferCreateInfo buffer_info = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(SceneUniforms), .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE, };
		VmaAllocationCreateInfo alloc_info = { .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, .usage = VMA_MEMORY_USAGE_AUTO, };
		VmaAllocationInfo allocation_info;
		if (vmaCreateBuffer(context.allocator, &buffer_info, &alloc_info, &vk_uniform_buffer_scene, &vk_uniform_buffer_scene_allocation, &allocation_info) != VK_SUCCESS) { std::cerr << "Failed to create scene uniform buffer\n"; return false; }
		vk_uniform_buffer_scene_mapped = allocation_info.pMappedData;

		constexpr uint32_t MAX_MODELS = 16;
		const size_t alignment = std::max<size_t>(1, static_cast<size_t>(context.min_uniform_buffer_offset_alignment));
		const size_t alignedSize = (sizeof(ModelUniforms) + alignment - 1) / alignment * alignment;
		g_model_stride = alignedSize;

		buffer_info.size = alignedSize * MAX_MODELS;
		if (vmaCreateBuffer(context.allocator, &buffer_info, &alloc_info, &vk_uniform_buffer_model, &vk_uniform_buffer_model_allocation, &allocation_info) != VK_SUCCESS) { std::cerr << "Failed to create model uniform buffer\n"; return false; }
		vk_uniform_buffer_model_mapped = allocation_info.pMappedData;

		buffer_info.size = sizeof(PointLight) * MAX_POINT_LIGHTS;
		buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		if (vmaCreateBuffer(context.allocator, &buffer_info, &alloc_info, &vk_storage_buffer_lights, &vk_storage_buffer_lights_allocation, &allocation_info) != VK_SUCCESS) { std::cerr << "Failed to create lights storage buffer\n"; return false; }
		vk_storage_buffer_lights_mapped = allocation_info.pMappedData;

		buffer_info.size = sizeof(SpotLight) * MAX_SPOT_LIGHTS;
		if (vmaCreateBuffer(context.allocator, &buffer_info, &alloc_info, &vk_storage_buffer_spot_lights, &vk_storage_buffer_spot_lights_allocation, &allocation_info) != VK_SUCCESS) { std::cerr << "Failed to create spot lights storage buffer\n"; return false; }
		vk_storage_buffer_spot_lights_mapped = allocation_info.pMappedData;

		VkDescriptorSetLayoutBinding scene_bindings[] = {
			{.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, },
			{.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, },
			{.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT, },
			{.binding = 3, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
		};
		VkDescriptorSetLayoutCreateInfo layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 4, .pBindings = scene_bindings, };
		if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &vk_descriptor_set_layout_scene) != VK_SUCCESS) { std::cerr << "Failed to create scene descriptor set layout\n"; return false; }

		VkDescriptorSetLayoutBinding model_binding = {
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		};
		layout_info.bindingCount = 1; layout_info.pBindings = &model_binding;
		if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &vk_descriptor_set_layout_model) != VK_SUCCESS) { std::cerr << "Failed to create model descriptor set layout\n"; return false; }

		VkDescriptorSetLayoutBinding tex_binding = { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT };
		VkDescriptorSetLayoutCreateInfo tex_layout_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &tex_binding };
		if (vkCreateDescriptorSetLayout(context.device, &tex_layout_info, nullptr, &vk_descriptor_set_layout_texture) != VK_SUCCESS) { std::cerr << "Failed to create texture descriptor set layout\n"; return false; }

		VkDescriptorPoolSize pool_sizes[] = {
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 },
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1 },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
			{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 9 },
		};
		VkDescriptorPoolCreateInfo pool_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 11, .poolSizeCount = 4, .pPoolSizes = pool_sizes, };
		if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &vk_descriptor_pool) != VK_SUCCESS) { std::cerr << "Failed to create descriptor pool\n"; return false; }

		VkDescriptorSetAllocateInfo alloc_set_info = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = vk_descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &vk_descriptor_set_layout_scene, };
		if (vkAllocateDescriptorSets(context.device, &alloc_set_info, &vk_descriptor_set_scene) != VK_SUCCESS) { std::cerr << "Failed to allocate scene descriptor set\n"; return false; }
		alloc_set_info.pSetLayouts = &vk_descriptor_set_layout_model;
		if (vkAllocateDescriptorSets(context.device, &alloc_set_info, &vk_descriptor_set_model) != VK_SUCCESS) { std::cerr << "Failed to allocate model descriptor set\n"; return false; }

		VkDescriptorBufferInfo scene_ubo_info   = { vk_uniform_buffer_scene,       0, sizeof(SceneUniforms) };
		VkDescriptorBufferInfo lights_info      = { vk_storage_buffer_lights,      0, VK_WHOLE_SIZE };
		VkDescriptorBufferInfo spot_lights_info = { vk_storage_buffer_spot_lights, 0, VK_WHOLE_SIZE };

		VkDescriptorImageInfo shadow_info{
			.sampler     = vk_sampler_shadow,
			.imageView   = vk_image_view_shadow_map,
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		};

		VkWriteDescriptorSet scene_writes[] = {
			{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_descriptor_set_scene,
			.dstBinding = 0, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.pBufferInfo = &scene_ubo_info },

			{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_descriptor_set_scene,
			.dstBinding = 1, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &lights_info },

			{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_descriptor_set_scene,
			.dstBinding = 2, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.pBufferInfo = &spot_lights_info },

			{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_descriptor_set_scene,
			.dstBinding = 3, .descriptorCount = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &shadow_info },
		};
		vkUpdateDescriptorSets(context.device, 4, scene_writes, 0, nullptr);

		VkDescriptorBufferInfo model_ubo_info = { vk_uniform_buffer_model, 0, sizeof(ModelUniforms) };
		VkWriteDescriptorSet model_write = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = vk_descriptor_set_model, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &model_ubo_info, };
		vkUpdateDescriptorSets(context.device, 1, &model_write, 0, nullptr);

		return true;
	}

	VkShaderModule loadShaderModule(const char* path) {
		auto fullPath = std::filesystem::absolute(path);
		std::cout << "Loading shader: " << fullPath << '\n';
		std::ifstream file(fullPath, std::ios::binary | std::ios::ate);
		if (!file) { std::cerr << "Failed to open shader file: " << path << '\n'; return VK_NULL_HANDLE; }
		const size_t size = file.tellg();
		std::vector<char> buffer(size);
		file.seekg(0); file.read(buffer.data(), size); file.close();
		const VkShaderModuleCreateInfo info = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = reinterpret_cast<const uint32_t*>(buffer.data()), };
		VkShaderModule module;
		if (vkCreateShaderModule(context.device, &info, nullptr, &module) != VK_SUCCESS) { std::cerr << "Failed to create shader module\n"; return VK_NULL_HANDLE; }
		return module;
	}

	bool createPipeline() {
		VkShaderModule vert_shader = loadShaderModule("shaders/shader.vert.spv");
		VkShaderModule frag_shader = loadShaderModule("shaders/shader.frag.spv");
		if (vert_shader == VK_NULL_HANDLE || frag_shader == VK_NULL_HANDLE) { return false; }
		const VkPipelineShaderStageCreateInfo stages[] = {
			{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vert_shader, .pName = "main", },
			{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag_shader, .pName = "main", },
		};
		const VkVertexInputBindingDescription binding = {
            .binding = 0,
            .stride = sizeof(float) * 8,
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

            {
                .location = 2,
                .binding = 0,
                .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = sizeof(float) * 6,
            },
        };

        const VkPipelineVertexInputStateCreateInfo vertex_input = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding,
            .vertexAttributeDescriptionCount = 3,
            .pVertexAttributeDescriptions = attributes,
        };
		const VkPipelineInputAssemblyStateCreateInfo input_assembly = { .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, };
		const VkPipelineViewportStateCreateInfo viewport_state = { .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1, };
		const VkPipelineRasterizationStateCreateInfo rasterizer = { .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .frontFace = VK_FRONT_FACE_CLOCKWISE, .lineWidth = 1.0f, };
		const VkPipelineMultisampleStateCreateInfo multisampling = { .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT, };
		const VkPipelineDepthStencilStateCreateInfo depth_stencil = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_LESS, };
		const VkPipelineColorBlendAttachmentState color_blend_attachment = { .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT, };
		const VkPipelineColorBlendStateCreateInfo color_blending = { .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &color_blend_attachment, };
		const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, };
		const VkPipelineDynamicStateCreateInfo dynamic_state = { .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 2, .pDynamicStates = dynamic_states, };
		VkDescriptorSetLayout set_layouts[] = {
			vk_descriptor_set_layout_scene,
			vk_descriptor_set_layout_model,
			vk_descriptor_set_layout_texture,
		};
		VkPushConstantRange push_range{};
		push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		push_range.offset = 0;
		push_range.size = sizeof(float) + sizeof(int) + sizeof(float) + sizeof(float);

		const VkPipelineLayoutCreateInfo layout_info = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 3,
			.pSetLayouts = set_layouts,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &push_range,
		};
		if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &vk_pipeline_layout) != VK_SUCCESS) { std::cerr << "Failed to create pipeline layout\n"; return false; }
		const VkGraphicsPipelineCreateInfo pipeline_info = {
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = stages,
			.pVertexInputState = &vertex_input, .pInputAssemblyState = &input_assembly, .pViewportState = &viewport_state,
			.pRasterizationState = &rasterizer, .pMultisampleState = &multisampling, .pDepthStencilState = &depth_stencil,
			.pColorBlendState = &color_blending, .pDynamicState = &dynamic_state, .layout = vk_pipeline_layout,
			.renderPass = context.render_pass, .subpass = 0,
		};
		if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &vk_pipeline) != VK_SUCCESS) { std::cerr << "Failed to create graphics pipeline\n"; return false; }

		VkShaderModule shadow_vert = loadShaderModule("shaders/shader_shadow.vert.spv");
		if (shadow_vert == VK_NULL_HANDLE) { return false; }

		VkPipelineShaderStageCreateInfo shadow_stage{
			VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
		shadow_stage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
		shadow_stage.module = shadow_vert;
		shadow_stage.pName  = "main";

		const VkPipelineRasterizationStateCreateInfo shadow_raster = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_NONE,
			.frontFace = VK_FRONT_FACE_CLOCKWISE,
			.depthBiasEnable = VK_TRUE,
			.lineWidth = 1.0f,
		};
		const VkPipelineDepthStencilStateCreateInfo shadow_depth = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
			.depthWriteEnable = VK_TRUE,
			.depthCompareOp = VK_COMPARE_OP_LESS,
		};
		const VkDynamicState shadow_dyn_states[] = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR,
			VK_DYNAMIC_STATE_DEPTH_BIAS,
		};
		const VkPipelineDynamicStateCreateInfo shadow_dyn = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
			.dynamicStateCount = 3,
			.pDynamicStates = shadow_dyn_states,
		};

		VkPushConstantRange shadow_pcr{
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
			.offset = 0,
			.size = sizeof(float) * 16,
		};
		VkPipelineLayoutCreateInfo shadow_layout{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &vk_descriptor_set_layout_model,
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &shadow_pcr,
		};
		if (vkCreatePipelineLayout(context.device, &shadow_layout, nullptr, &vk_pipeline_layout_shadow) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow pipeline layout\n";
			vkDestroyShaderModule(context.device, shadow_vert, nullptr); return false;
		}

		VkPipelineRenderingCreateInfo shadow_ri{
			VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
		shadow_ri.colorAttachmentCount = 0;
		shadow_ri.pColorAttachmentFormats = nullptr;
		shadow_ri.depthAttachmentFormat = SHADOW_MAP_FORMAT;

		VkGraphicsPipelineCreateInfo shadow_pi{
			VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
		shadow_pi.pNext = &shadow_ri;
		shadow_pi.stageCount = 1;
		shadow_pi.pStages = &shadow_stage;
		shadow_pi.pVertexInputState = &vertex_input;
		shadow_pi.pInputAssemblyState = &input_assembly;
		shadow_pi.pViewportState = &viewport_state;
		shadow_pi.pRasterizationState = &shadow_raster;
		shadow_pi.pMultisampleState = &multisampling;
		shadow_pi.pDepthStencilState = &shadow_depth;
		shadow_pi.pColorBlendState = nullptr;
		shadow_pi.pDynamicState = &shadow_dyn;
		shadow_pi.layout = vk_pipeline_layout_shadow;
		shadow_pi.renderPass = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &shadow_pi,
									nullptr, &vk_pipeline_shadow) != VK_SUCCESS) {
			std::cerr << "Failed to create shadow pipeline\n";
			vkDestroyPipelineLayout(context.device, vk_pipeline_layout_shadow, nullptr);
			vkDestroyShaderModule(context.device, shadow_vert, nullptr); return false;
		}
		vkDestroyShaderModule(context.device, shadow_vert, nullptr);

		vkDestroyShaderModule(context.device, vert_shader, nullptr);
		vkDestroyShaderModule(context.device, frag_shader, nullptr);
		return true;
	}

	void drawShadowPass(VkCommandBuffer cmd) {
		VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
		b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		b.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
		b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		b.image = vk_image_shadow_map;
		b.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
		b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
			VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
			0, 0, nullptr, 0, nullptr, 1, &b);

		VkRenderingAttachmentInfo depth_att{
			VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
		depth_att.imageView = vk_image_view_shadow_map;
		depth_att.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
		depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depth_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		depth_att.clearValue.depthStencil = { 1.0f, 0 };

		VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
		ri.renderArea = { {0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE} };
		ri.layerCount = 1;
		ri.colorAttachmentCount = 0;
		ri.pDepthAttachment = &depth_att;

		vkCmdBeginRendering(cmd, &ri);

		VkViewport vp{ 0, 0, float(SHADOW_MAP_SIZE), float(SHADOW_MAP_SIZE), 0, 1 };
		VkRect2D sc{ {0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE} };
		vkCmdSetViewport(cmd, 0, 1, &vp);
		vkCmdSetScissor (cmd, 0, 1, &sc);
		vkCmdSetDepthBias(cmd, g_depthBiasConstant, 0.0f, g_depthBiasSlope);

		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_shadow);

		VkDeviceSize off = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &vk_vertex_buffer, &off);
		vkCmdBindIndexBuffer (cmd, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdPushConstants(cmd, vk_pipeline_layout_shadow,
			VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(g_lightViewProj), g_lightViewProj);

		for (int i = 0; i < g_drawModelCount; ++i) {
			uint32_t dyn_off = uint32_t(g_model_stride * i);
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
				vk_pipeline_layout_shadow, 0, 1, &vk_descriptor_set_model, 1, &dyn_off);
			vkCmdDrawIndexed(cmd, 36, 1, 0, 0, 0);
		}

		vkCmdEndRendering(cmd);

		VkImageMemoryBarrier b2 = b;
		b2.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
		b2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		b2.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		b2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		vkCmdPipelineBarrier(cmd,
			VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0, 0, nullptr, 0, nullptr, 1, &b2);
	}

	void drawCube(VkCommandBuffer command_buffer) {
		vkResetCommandBuffer(command_buffer, 0);
		const VkCommandBufferBeginInfo begin_info = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, };
		vkBeginCommandBuffer(command_buffer, &begin_info);
		const VkClearValue clear_values[] = { {.color = {.float32 = { 0.1f, 0.1f, 0.1f, 1.0f } } }, {.depthStencil = { 1.0f, 0 } }, };
		const VkRenderPassBeginInfo render_pass_begin = {
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = context.render_pass,
			.framebuffer = vk_framebuffers[vk_swapchain_current_image], .renderArea = {.extent = context.swapchain_extent },
			.clearValueCount = 2, .pClearValues = clear_values,
		};
		drawShadowPass(command_buffer);
		vkCmdBeginRenderPass(command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);
		const VkViewport viewport = { .x = 0, .y = 0, .width = float(context.swapchain_extent.width), .height = float(context.swapchain_extent.height), .minDepth = 0, .maxDepth = 1, };
		const VkRect2D scissor = { .extent = context.swapchain_extent };
		vkCmdSetViewport(command_buffer, 0, 1, &viewport);
		vkCmdSetScissor(command_buffer, 0, 1, &scissor);
		vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);
		const VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(command_buffer, 0, 1, &vk_vertex_buffer, &offset);
		vkCmdBindIndexBuffer(command_buffer, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);
		vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
			vk_pipeline_layout, 0, 1, &vk_descriptor_set_scene, 0, nullptr);

		if (!g_textures.empty()) {
			int idx = std::clamp(g_activeTextureIndex, 0, (int)g_textures.size() - 1);
			VkDescriptorSet texSet = g_textures[idx].descriptorSet;
			vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				vk_pipeline_layout, 2, 1, &texSet, 0, nullptr);
		}

		struct PushConstants {
			float time;
			int   enableWeirdSampling;
			float shadowBias;
			float shadowTexelSize;
		};
		PushConstants push{};
		push.time                = g_pushTime;
		push.enableWeirdSampling = g_pushEnableWeirdSampling;
		push.shadowBias          = g_pushShadowBias;
		push.shadowTexelSize     = 1.0f / float(SHADOW_MAP_SIZE);
		vkCmdPushConstants(command_buffer, vk_pipeline_layout,
			VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &push);

		for (int i = 0; i < g_drawModelCount; ++i) {
			uint32_t dyn_off = uint32_t(g_model_stride * i);
			vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				vk_pipeline_layout, 1, 1, &vk_descriptor_set_model, 1, &dyn_off);
			vkCmdDrawIndexed(command_buffer, 36, 1, 0, 0, 0);
		}
		vkCmdEndRenderPass(command_buffer);
		vkEndCommandBuffer(command_buffer);
	}

	void updateSceneUniformBuffer(
		const float* viewProj,
		const float* cameraPos,
		int pointLightCount,
		const float* ambient,
		int spotLightCount,
		const float* directionalDir,
		const float* directionalColor,
		float directionalIntensity,
		const float* lightViewProj)
	{
		SceneUniforms u{};
		memcpy(u.viewProj, viewProj, sizeof(u.viewProj));

		u.cameraPos[0] = cameraPos[0];
		u.cameraPos[1] = cameraPos[1];
		u.cameraPos[2] = cameraPos[2];
		u.cameraPos[3] = 0.0f;

		u.ambientColor[0] = ambient[0];
		u.ambientColor[1] = ambient[1];
		u.ambientColor[2] = ambient[2];
		u.ambientColor[3] = 0.0f;

		u.directionalDirection[0] = directionalDir[0];
		u.directionalDirection[1] = directionalDir[1];
		u.directionalDirection[2] = directionalDir[2];
		u.directionalDirection[3] = 0.0f;

		u.directionalColor[0] = directionalColor[0];
		u.directionalColor[1] = directionalColor[1];
		u.directionalColor[2] = directionalColor[2];
		u.directionalColor[3] = directionalIntensity;
		memcpy(u.lightViewProj, lightViewProj, sizeof(u.lightViewProj));

		u.pointLightCount = pointLightCount;
		u.spotLightCount = spotLightCount;

		memcpy(vk_uniform_buffer_scene_mapped, &u, sizeof(SceneUniforms));
	}

	void updateModelUniformBuffer(
		uint32_t slot,
		const float* model,
		const float* albedo,
		float shininess,
		const float* specColor,
		int useTexture)
	{
		ModelUniforms u{};
		memcpy(u.model, model, sizeof(u.model));
		u.albedo[0] = albedo[0];
		u.albedo[1] = albedo[1];
		u.albedo[2] = albedo[2];
		u.shininess = shininess;
		u.specularColor[0] = specColor[0];
		u.specularColor[1] = specColor[1];
		u.specularColor[2] = specColor[2];
		u.useTexture = useTexture;

		char* dst = static_cast<char*>(vk_uniform_buffer_model_mapped) + g_model_stride * slot;
		memcpy(dst, &u, sizeof(ModelUniforms));
	}

	void updateLights(const PointLight* lights, int count) {
		memcpy(vk_storage_buffer_lights_mapped, lights, sizeof(PointLight) * count);
	}

	void updateSpotLights(const SpotLight* lights, int count) {
		memcpy(vk_storage_buffer_spot_lights_mapped, lights, sizeof(SpotLight) * count);
	}

	bool initialize(GLFWwindow* const window) {
		vkb::InstanceBuilder ib;
		auto ibr = ib.require_api_version(VK_MAKE_VERSION(1, 3, 0)).request_validation_layers().build();
		auto vkb_instance = ibr.value();
		vk_instance = vkb_instance.instance;
		vk_api_version = vkb_instance.api_version;
		if (glfwCreateWindowSurface(vk_instance, window, nullptr, &vk_surface) != VK_SUCCESS) {
			const char* message = nullptr;
			glfwGetError(&message);
			std::cerr << message << '\n';
			return false;
		}
		VkPhysicalDeviceVulkan13Features features13{};
		features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
		features13.dynamicRendering = VK_TRUE;

		vkb::PhysicalDeviceSelector pds(vkb_instance, vk_surface);
		auto pds_result = pds.set_minimum_version(1, 3)
							.set_required_features_13(features13)
							.prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
							.require_present()
							.select();
		if (!pds_result) { std::cerr << pds_result.error().message() << '\n'; return false; }
		auto vkb_physical_device = pds_result.value();
		vk_depth_buffer_format = selectDepthFormat(vkb_physical_device.physical_device);
		if (vk_depth_buffer_format == VK_FORMAT_UNDEFINED) { std::cerr << "No supported depth/stencil attachment format found\n"; return false; }
		vkb::DeviceBuilder db(vkb_physical_device);
		auto db_result = db.build();
		if (!db_result) { std::cerr << db_result.error().message() << '\n'; return false; }
		auto vkb_device = db_result.value();
		context.physical_device = vkb_device.physical_device;
		context.device = vkb_device.device;
		VkPhysicalDeviceProperties props{};
		vkGetPhysicalDeviceProperties(context.physical_device, &props);
		context.min_uniform_buffer_offset_alignment = props.limits.minUniformBufferOffsetAlignment;
		if (auto result = vkb_device.get_queue(vkb::QueueType::graphics); result) { context.graphics_queue = result.value(); }
		else { std::cerr << result.error().message() << '\n'; return false; }
		if (auto result = vkb_device.get_queue_index(vkb::QueueType::graphics); result) { context.graphics_queue_index = result.value(); }
		else { std::cerr << result.error().message() << '\n'; return false; }
		const VmaAllocatorCreateInfo allocator = { .physicalDevice = context.physical_device, .device = context.device, .instance = vk_instance, .vulkanApiVersion = vk_api_version, };
		if (vmaCreateAllocator(&allocator, &context.allocator) != VK_SUCCESS) { std::cerr << "Failed to create Vulkan Memory Allocator\n"; return false; }
		vkb::SwapchainBuilder sb(vkb_device);
		auto sb_result = sb.use_default_format_selection().set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR).use_default_image_usage_flags().build();
		if (!sb_result) { std::cerr << sb_result.error().message() << '\n'; return false; }
		auto vkb_swapchain = sb_result.value();
		vk_swapchain = vkb_swapchain.swapchain;
		context.swapchain_format = vkb_swapchain.image_format;
		context.swapchain_extent = vkb_swapchain.extent;
		vk_swapchain_images = vkb_swapchain.get_images().value();
		vk_swapchain_image_views = vkb_swapchain.get_image_views().value();
		vk_swapchain_current_image = UINT32_MAX;
		const uint32_t swapchain_images_count = uint32_t(vk_swapchain_images.size());
		const VkImageCreateInfo depth_buffer = { .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = vk_depth_buffer_format, .extent = { context.swapchain_extent.width, context.swapchain_extent.height, 1 }, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, };
		const VmaAllocationCreateInfo depth_buffer_allocation = { .usage = VMA_MEMORY_USAGE_AUTO, };
		if (vmaCreateImage(context.allocator, &depth_buffer, &depth_buffer_allocation, &vk_image_depth_buffer, &vma_allocation_depth_buffer, nullptr) != VK_SUCCESS) { std::cerr << "Failed to allocate and create Vulkan image for depth buffer\n"; return false; }
		const VkImageViewCreateInfo depth_buffer_view = { .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = vk_image_depth_buffer, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = vk_depth_buffer_format, .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1, }, };
		if (vkCreateImageView(context.device, &depth_buffer_view, nullptr, &vk_image_view_depth_buffer) != VK_SUCCESS) { std::cerr << "Failed to create Vulkan image view for depth buffer\n"; return false; }
		const VkAttachmentDescription render_pass_attachments[] = {
			{.format = context.swapchain_format, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, },
			{.format = vk_depth_buffer_format, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, },
		};
		const VkAttachmentReference render_pass_color_attachment = { .attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, };
		const VkAttachmentReference render_pass_depth_attachment = { .attachment = 1, .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, };
		const VkSubpassDescription render_pass_subpass = { .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1, .pColorAttachments = &render_pass_color_attachment, .pDepthStencilAttachment = &render_pass_depth_attachment, };
		const VkRenderPassCreateInfo render_pass = { .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = sizeof(render_pass_attachments) / sizeof(render_pass_attachments[0]), .pAttachments = render_pass_attachments, .subpassCount = 1, .pSubpasses = &render_pass_subpass, };
		if (vkCreateRenderPass(context.device, &render_pass, nullptr, &context.render_pass) != VK_SUCCESS) { std::cerr << "Failed to create Vulkan render pass\n"; return false; }
		VkImageView framebuffer_attachments[] = { VK_NULL_HANDLE, vk_image_view_depth_buffer };
		const VkFramebufferCreateInfo framebuffer = { .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = context.render_pass, .attachmentCount = sizeof(framebuffer_attachments) / sizeof(framebuffer_attachments[0]), .pAttachments = framebuffer_attachments, .width = context.swapchain_extent.width, .height = context.swapchain_extent.height, .layers = 1, };
		vk_framebuffers.resize(swapchain_images_count);
		for (uint32_t i = 0; i < swapchain_images_count; ++i) {
			framebuffer_attachments[0] = vk_swapchain_image_views[i];
			if (vkCreateFramebuffer(context.device, &framebuffer, nullptr, &vk_framebuffers[i]) != VK_SUCCESS) { std::cerr << "Failed to create Vulkan framebuffer #" << i << '\n'; return false; }
		}
		const VkSemaphoreCreateInfo semaphore = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		const VkFenceCreateInfo fence = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT, };
		vk_semaphores_image_finished.resize(swapchain_images_count);
		for (uint32_t i = 0; i < swapchain_images_count; ++i) {
			if (vkCreateSemaphore(context.device, &semaphore, nullptr, &vk_semaphores_image_finished[i]) != VK_SUCCESS) { std::cerr << "Failed to create Vulkan semaphore #" << i << " for finished image\n"; return false; }
		}
		if (vkCreateSemaphore(context.device, &semaphore, NULL, &vk_semaphore_image_available) != VK_SUCCESS) { std::cerr << "Failed to create Vulkan semaphore for available image\n"; return false; }
		if (vkCreateFence(context.device, &fence, nullptr, &vk_fence_frame_in_flight) != VK_SUCCESS) { std::cerr << "Failed to create Vulkan semaphore for image in flight\n"; return false; }
		const VkCommandPoolCreateInfo command_pool = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = context.graphics_queue_index, };
		if (vkCreateCommandPool(context.device, &command_pool, nullptr, &vk_command_pool) != VK_SUCCESS) { std::cerr << "Failed to create Vulkan command pool\n"; return false; }
		const VkCommandBufferAllocateInfo command_buffers = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = vk_command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1, };
		if (vkAllocateCommandBuffers(context.device, &command_buffers, &vk_command_buffer) != VK_SUCCESS) { std::cerr << "Failed to allocate Vulkan command buffer\n"; return false; }
		if (!initializeImGUI()) { std::cerr << "Failed to initialize ImGUI Vulkan rendering backend\n"; return false; }
		return true;
	}

	void shutdown() {
		vkQueueWaitIdle(context.graphics_queue);
		ImGui_ImplVulkan_Shutdown();
		vkDestroyCommandPool(context.device, vk_imgui_command_pool, nullptr);
		for (size_t i = 0, n = vk_imgui_framebuffers.size(); i < n; ++i) { vkDestroyFramebuffer(context.device, vk_imgui_framebuffers[i], nullptr); }
		vkDestroyRenderPass(context.device, vk_imgui_render_pass, nullptr);
		vkDestroyDescriptorPool(context.device, vk_imgui_descriptor_pool, nullptr);
		vkDestroyCommandPool(context.device, vk_command_pool, nullptr);
		vkDestroyFence(context.device, vk_fence_frame_in_flight, nullptr);
		for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) { vkDestroySemaphore(context.device, vk_semaphores_image_finished[i], nullptr); }
		vkDestroySemaphore(context.device, vk_semaphore_image_available, nullptr);
		for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) { vkDestroyFramebuffer(context.device, vk_framebuffers[i], nullptr); }
		vkDestroyRenderPass(context.device, context.render_pass, nullptr);
		vkDestroyImageView(context.device, vk_image_view_depth_buffer, nullptr);
		vmaDestroyImage(context.allocator, vk_image_depth_buffer, vma_allocation_depth_buffer);
		for (size_t i = 0, n = vk_swapchain_images.size(); i < n; ++i) { vkDestroyImageView(context.device, vk_swapchain_image_views[i], nullptr); }
		vkDestroySwapchainKHR(context.device, vk_swapchain, nullptr);
		vkDestroyPipeline(context.device, vk_pipeline, nullptr);
		vkDestroyPipelineLayout(context.device, vk_pipeline_layout, nullptr);
		vkDestroyPipeline      (context.device, vk_pipeline_shadow, nullptr);
		vkDestroyPipelineLayout(context.device, vk_pipeline_layout_shadow, nullptr);
		vkDestroySampler       (context.device, vk_sampler_shadow, nullptr);
		vkDestroyImageView     (context.device, vk_image_view_shadow_map, nullptr);
		vmaDestroyImage        (context.allocator, vk_image_shadow_map, vma_allocation_shadow_map);
		vkDestroyDescriptorPool(context.device, vk_descriptor_pool, nullptr);
		vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout_scene, nullptr);
		vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout_model, nullptr);
		vkDestroyDescriptorSetLayout(context.device, vk_descriptor_set_layout_texture, nullptr);
		vmaDestroyBuffer(context.allocator, vk_vertex_buffer, vk_vertex_buffer_allocation);
		vmaDestroyBuffer(context.allocator, vk_index_buffer, vk_index_buffer_allocation);
		vmaDestroyBuffer(context.allocator, vk_uniform_buffer_scene, vk_uniform_buffer_scene_allocation);
		vmaDestroyBuffer(context.allocator, vk_uniform_buffer_model, vk_uniform_buffer_model_allocation);
		vmaDestroyBuffer(context.allocator, vk_storage_buffer_lights, vk_storage_buffer_lights_allocation);
		vmaDestroyBuffer(context.allocator, vk_storage_buffer_spot_lights, vk_storage_buffer_spot_lights_allocation);
		for (auto& tex : g_textures) {
			vkDestroySampler(context.device, tex.sampler, nullptr);
			vkDestroyImageView(context.device, tex.view, nullptr);
			vmaDestroyImage(context.allocator, tex.image, tex.allocation);
		}
		g_textures.clear();
		vmaDestroyAllocator(context.allocator);
		vkDestroyDevice(context.device, nullptr);
		vkDestroySurfaceKHR(vk_instance, vk_surface, nullptr);
		vkDestroyInstance(vk_instance, nullptr);
	}

	void resize(uint32_t width, uint32_t height) {
		if (width == 0 || height == 0) { return; }
		vk_swapchain_resize_width = width;
		vk_swapchain_resize_height = height;
		vk_swapchain_resize_require = true;
	}

	FrameData prepare() {
		vkWaitForFences(context.device, 1, &vk_fence_frame_in_flight, VK_TRUE, UINT64_MAX);
	retry_acquire:
		switch (vkAcquireNextImageKHR(context.device, vk_swapchain, UINT64_MAX, vk_semaphore_image_available, VK_NULL_HANDLE, &vk_swapchain_current_image)) {
		case VK_SUCCESS: break;
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
		return { .framebuffer = vk_framebuffers[vk_swapchain_current_image], .command_buffer = vk_command_buffer, };
	}

	void submitAndPresent() {
		drawImGUI();
		const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		const VkCommandBuffer command_buffers[] = { vk_command_buffer, vk_imgui_command_buffer, };
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

            rebuildSwapchain(
                vk_swapchain_resize_width,
                vk_swapchain_resize_height
            );
        }
        else if (result != VK_SUCCESS) {
            std::cerr << "Failed to present Vulkan swapchain image\n";
        }
	}

} // namespace graphics::internal