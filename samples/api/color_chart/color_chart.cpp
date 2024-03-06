/* Copyright (c) 2023, Arm Limited and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 the "License";
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "color_chart.h"

const uint32_t CUBE_SIZE            = 24;
const uint32_t COLUMN_COUNT         = 6;
const uint32_t ROW_COUNT            = 4;
const uint32_t PADDING              = 1;
const uint32_t SAMPLE_WIDTH         = (CUBE_SIZE + PADDING) * COLUMN_COUNT + PADDING;
const uint32_t SAMPLE_HEIGHT        = (CUBE_SIZE + PADDING) * ROW_COUNT + PADDING;
const uint32_t SAVE_WIDTH           = 2880;        // has to be multiple of 32 for stbi_write_png to be properly aligned, idk why
const uint32_t SAVE_HEIGHT          = 1920;
const VkFormat SAMPLE_FORMAT        = VK_FORMAT_R8G8B8A8_UNORM;        // VK_FORMAT_R32G32B32A32_SFLOAT;
const VkFormat SAVE_FORMAT          = VK_FORMAT_R8G8B8A8_UNORM;        // VK_FORMAT_R32G32B32A32_SFLOAT;
const uint32_t SAVE_COMPONENTS      = 4;
const char    *SAVED_IMAGE_FILENAME = "color_chart";
const bool     DRAW_UI              = false;
const float    MAX_TIME             = 5.0f;        // seconds

struct PushConstant
{
	float time;        // seconds
	float dt;          // seconds
};

VkVertexInputBindingDescription ColoredVertex2D::getBindingDescription()
{
	return vkb::initializers::vertex_input_binding_description(0, sizeof(ColoredVertex2D), VK_VERTEX_INPUT_RATE_VERTEX);
}

std::array<VkVertexInputAttributeDescription, 2> ColoredVertex2D::getAttributeDescriptions()
{
	return {
	    vkb::initializers::vertex_input_attribute_description(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ColoredVertex2D, pos)),
	    vkb::initializers::vertex_input_attribute_description(0, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ColoredVertex2D, color))};
}

color_chart::color_chart()
{
}

color_chart::~color_chart()
{
	if (device)
	{
		vkDestroyDescriptorPool(get_device().get_handle(), descriptor_pool, nullptr);

		vkDestroySampler(get_device().get_handle(), textureSampler, nullptr);

		vkDestroyBuffer(get_device().get_handle(), indexBuffer, nullptr);
		vkFreeMemory(get_device().get_handle(), indexBufferMemory, nullptr);

		vkDestroyBuffer(get_device().get_handle(), vertexBuffer, nullptr);
		vkFreeMemory(get_device().get_handle(), vertexBufferMemory, nullptr);

		vkDestroyImageView(get_device().get_handle(), savedImageView, nullptr);
		vkDestroyImage(get_device().get_handle(), savedImage, nullptr);
		vkFreeMemory(get_device().get_handle(), savedImageMemory, nullptr);

		vkDestroyFramebuffer(get_device().get_handle(), saved_framebuffer, nullptr);

		for (const auto &view : textureImageViews)
		{
			vkDestroyImageView(get_device().get_handle(), view, nullptr);
		}
		for (const auto &image : textureImages)
		{
			vkDestroyImage(get_device().get_handle(), image, nullptr);
		}
		for (const auto &textureImageMemory : textureImageMemories)
		{
			vkFreeMemory(get_device().get_handle(), textureImageMemory, nullptr);
		}

		for (const auto &framebuffer : sample_framebuffers)
		{
			vkDestroyFramebuffer(get_device().get_handle(), framebuffer, nullptr);
		}

		vkDestroyPipeline(get_device().get_handle(), save_pipeline, nullptr);

		vkDestroyPipeline(get_device().get_handle(), upsample_pipeline, nullptr);
		vkDestroyPipelineLayout(get_device().get_handle(), upsample_pipeline_layout, nullptr);

		vkDestroyPipeline(get_device().get_handle(), sample_pipeline, nullptr);
		vkDestroyPipelineLayout(get_device().get_handle(), sample_pipeline_layout, nullptr);

		vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptorSetLayout, nullptr);

		vkDestroyRenderPass(get_device().get_handle(), save_render_pass, nullptr);
		vkDestroyRenderPass(get_device().get_handle(), sample_render_pass, nullptr);

		vkDestroyFence(get_device().get_handle(), savedFence, nullptr);
	}
}

bool color_chart::prepare(const vkb::ApplicationOptions &options)
{
	// Add support for wide color gamut
	add_instance_extension(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);

	if (!ApiVulkanSample::prepare(options))
	{
		return false;
	}

	createDescriptorSetLayout();
	prepare_pipelines();
	cmd_buffer_count = draw_cmd_buffers.size() + 1;        // last cmd buffer is for saving image
	textureImages.resize(cmd_buffer_count);
	textureImageMemories.resize(cmd_buffer_count);
	textureImageViews.resize(cmd_buffer_count);
	sample_framebuffers.resize(cmd_buffer_count);
	for (uint32_t i = 0; i < cmd_buffer_count; ++i)
	{
		// Render target of sample pass
		createImage(
		    SAMPLE_WIDTH,
		    SAMPLE_HEIGHT,
		    SAMPLE_FORMAT,
		    VK_IMAGE_TILING_OPTIMAL,
		    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		    textureImages[i],
		    textureImageMemories[i]);
		createTextureImageView(textureImages[i], textureImageViews[i]);
		createSampleFramebuffer(textureImageViews[i], sample_framebuffers[i]);
	}
	// Render target of save pass
	createImage(
	    SAVE_WIDTH,
	    SAVE_HEIGHT,
	    SAVE_FORMAT,
	    VK_IMAGE_TILING_LINEAR,
	    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
	    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
	    savedImage,
	    savedImageMemory);
	createTextureImageView(savedImage, savedImageView);
	createSavedFramebuffer();
	createGeometry();
	createTextureSampler();
	createDescriptorPool();
	createDescriptorSets();
	createSaveCommandBuffer();
	build_command_buffers();
	startTime = std::chrono::high_resolution_clock::now();
	prepared  = true;
	return true;
}

void color_chart::create_render_context()
{
	// Use wide color gamut hdr when possible
	auto surface_priority_list = std::vector<VkSurfaceFormatKHR>{{VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT},
	                                                             {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
	                                                             {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}};

	VulkanSample::create_render_context(surface_priority_list);
}

void color_chart::setup_render_pass()
{
	VkAttachmentDescription sample_attachment = {};
	// Color attachment
	sample_attachment.format         = SAMPLE_FORMAT;
	sample_attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
	sample_attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
	sample_attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	sample_attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	sample_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	sample_attachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	sample_attachment.finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	std::array<VkAttachmentDescription, 2> attachments = {};
	// Color attachment
	attachments[0].format         = render_context->get_format();
	attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
	// Depth attachment for compatibility with framework
	attachments[1].format         = depth_format;
	attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription save_attachment = {};
	// Color attachment
	save_attachment.format         = SAVE_FORMAT;
	save_attachment.samples        = VK_SAMPLE_COUNT_1_BIT;
	save_attachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
	save_attachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	save_attachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	save_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	save_attachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	save_attachment.finalLayout    = VK_IMAGE_LAYOUT_GENERAL;

	VkAttachmentReference color_reference = {};
	color_reference.attachment            = 0;
	color_reference.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depth_reference = {};
	depth_reference.attachment            = 1;
	depth_reference.layout                = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkSubpassDescription sample_subpass_description = {};
	sample_subpass_description.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
	sample_subpass_description.colorAttachmentCount = 1;
	sample_subpass_description.pColorAttachments    = &color_reference;

	VkSubpassDescription subpass_description    = {};
	subpass_description.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass_description.colorAttachmentCount    = 1;
	subpass_description.pColorAttachments       = &color_reference;
	subpass_description.pDepthStencilAttachment = &depth_reference;
	subpass_description.inputAttachmentCount    = 0;
	subpass_description.pInputAttachments       = nullptr;
	subpass_description.preserveAttachmentCount = 0;
	subpass_description.pPreserveAttachments    = nullptr;
	subpass_description.pResolveAttachments     = nullptr;

	// Subpass dependencies for layout transitions
	VkSubpassDependency dependency = {};
	dependency.srcSubpass          = VK_SUBPASS_EXTERNAL;
	dependency.dstSubpass          = 0;
	dependency.srcStageMask        = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
	dependency.dstStageMask        = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
	dependency.srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependency.dstAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
	dependency.dependencyFlags     = VK_DEPENDENCY_BY_REGION_BIT;

	VkRenderPassCreateInfo sample_render_pass_create_info = {};
	sample_render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	sample_render_pass_create_info.attachmentCount        = 1;
	sample_render_pass_create_info.pAttachments           = &sample_attachment;
	sample_render_pass_create_info.subpassCount           = 1;
	sample_render_pass_create_info.pSubpasses             = &sample_subpass_description;
	sample_render_pass_create_info.dependencyCount        = 1;
	sample_render_pass_create_info.pDependencies          = &dependency;

	VkRenderPassCreateInfo render_pass_create_info = vkb::initializers::render_pass_create_info();
	render_pass_create_info.attachmentCount        = static_cast<uint32_t>(attachments.size());
	render_pass_create_info.pAttachments           = attachments.data();
	render_pass_create_info.subpassCount           = 1;
	render_pass_create_info.pSubpasses             = &subpass_description;
	render_pass_create_info.dependencyCount        = 1;
	render_pass_create_info.pDependencies          = &dependency;

	VkRenderPassCreateInfo save_render_pass_create_info = vkb::initializers::render_pass_create_info();
	save_render_pass_create_info.attachmentCount        = 1;
	save_render_pass_create_info.pAttachments           = &save_attachment;
	save_render_pass_create_info.subpassCount           = 1;
	save_render_pass_create_info.pSubpasses             = &sample_subpass_description;
	save_render_pass_create_info.dependencyCount        = 1;
	save_render_pass_create_info.pDependencies          = &dependency;

	VK_CHECK(vkCreateRenderPass(get_device().get_handle(), &sample_render_pass_create_info, nullptr, &sample_render_pass));

	VK_CHECK(vkCreateRenderPass(get_device().get_handle(), &render_pass_create_info, nullptr, &render_pass));

	VK_CHECK(vkCreateRenderPass(get_device().get_handle(), &save_render_pass_create_info, nullptr, &save_render_pass));
}

void color_chart::input_event(const vkb::InputEvent &input_event)
{
	ApiVulkanSample::input_event(input_event);
	if (input_event.get_source() == vkb::EventSource::Keyboard)
	{
		const auto &key_button = static_cast<const vkb::KeyInputEvent &>(input_event);

		if (key_button.get_action() == vkb::KeyAction::Down)
		{
			switch (key_button.get_code())
			{
				case vkb::KeyCode::F2:
					exportImage();
					break;
				default:
					break;
			}
		}
	}
}

void color_chart::prepare_pipelines()
{
	// Sample pipeline
	VkPushConstantRange range = {};
	range.stageFlags          = VK_SHADER_STAGE_FRAGMENT_BIT;
	range.offset              = 0;
	range.size                = sizeof(PushConstant);

	VkPipelineLayoutCreateInfo layout_info = vkb::initializers::pipeline_layout_create_info(nullptr, 0);
	layout_info.pNext                      = nullptr;
	layout_info.flags                      = 0;
	layout_info.pushConstantRangeCount     = 1;
	layout_info.pPushConstantRanges        = &range;
	VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &layout_info, nullptr, &sample_pipeline_layout));

	VkPipelineVertexInputStateCreateInfo vertex_input = vkb::initializers::pipeline_vertex_input_state_create_info();

	auto bindingDescription    = ColoredVertex2D::getBindingDescription();
	auto attributeDescriptions = ColoredVertex2D::getAttributeDescriptions();

	vertex_input.vertexBindingDescriptionCount   = 1;
	vertex_input.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
	vertex_input.pVertexBindingDescriptions      = &bindingDescription;
	vertex_input.pVertexAttributeDescriptions    = attributeDescriptions.data();

	// Specify we will use triangle lists to draw geometry.
	VkPipelineInputAssemblyStateCreateInfo input_assembly = vkb::initializers::pipeline_input_assembly_state_create_info(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);

	// Specify rasterization state.
	VkPipelineRasterizationStateCreateInfo raster = vkb::initializers::pipeline_rasterization_state_create_info(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE);

	// Our attachment will write to all color channels, but no blending is enabled.
	VkPipelineColorBlendAttachmentState blend_attachment = vkb::initializers::pipeline_color_blend_attachment_state(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT, VK_FALSE);

	VkPipelineColorBlendStateCreateInfo blend = vkb::initializers::pipeline_color_blend_state_create_info(1, &blend_attachment);

	// We will have one viewport and scissor box.
	VkPipelineViewportStateCreateInfo viewport = vkb::initializers::pipeline_viewport_state_create_info(1, 1);

	// Disable depth testing (using reversed depth-buffer for increased precision).
	VkPipelineDepthStencilStateCreateInfo depth_stencil = vkb::initializers::pipeline_depth_stencil_state_create_info(VK_FALSE, VK_FALSE, VK_COMPARE_OP_ALWAYS);

	// No multisampling.
	VkPipelineMultisampleStateCreateInfo multisample = vkb::initializers::pipeline_multisample_state_create_info(VK_SAMPLE_COUNT_1_BIT);

	// Specify that these states will be dynamic, i.e. not part of pipeline state object.
	std::array<VkDynamicState, 2>    dynamics{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamic = vkb::initializers::pipeline_dynamic_state_create_info(dynamics.data(), vkb::to_u32(dynamics.size()));

	// Load our SPIR-V shaders.
	std::array<VkPipelineShaderStageCreateInfo, 2> shader_stages{};

	// Vertex stage of the pipeline
	shader_stages[0] = load_shader("shader.vert", VK_SHADER_STAGE_VERTEX_BIT);
	shader_stages[1] = load_shader("shader.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

	// We need to specify the pipeline layout and the render pass description up front as well.
	VkGraphicsPipelineCreateInfo pipeline_create_info = vkb::initializers::pipeline_create_info(sample_pipeline_layout, sample_render_pass);
	pipeline_create_info.stageCount                   = vkb::to_u32(shader_stages.size());
	pipeline_create_info.pStages                      = shader_stages.data();
	pipeline_create_info.pVertexInputState            = &vertex_input;
	pipeline_create_info.pInputAssemblyState          = &input_assembly;
	pipeline_create_info.pRasterizationState          = &raster;
	pipeline_create_info.pColorBlendState             = &blend;
	pipeline_create_info.pMultisampleState            = &multisample;
	pipeline_create_info.pViewportState               = &viewport;
	pipeline_create_info.pDepthStencilState           = &depth_stencil;
	pipeline_create_info.pDynamicState                = &dynamic;

	VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1, &pipeline_create_info, nullptr, &sample_pipeline));

	// Upsample pipeline
	VkPipelineLayoutCreateInfo upsample_layout_info = vkb::initializers::pipeline_layout_create_info(&descriptorSetLayout, 1);
	VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &upsample_layout_info, nullptr, &upsample_pipeline_layout));

	std::array<VkPipelineShaderStageCreateInfo, 2> upsample_shader_stages{};
	upsample_shader_stages[0] = load_shader("quad.vert", VK_SHADER_STAGE_VERTEX_BIT);
	upsample_shader_stages[1] = load_shader("texture.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

	vertex_input = vkb::initializers::pipeline_vertex_input_state_create_info();

	VkGraphicsPipelineCreateInfo upsample_pipeline_create_info = vkb::initializers::pipeline_create_info(upsample_pipeline_layout, render_pass);
	upsample_pipeline_create_info.stageCount                   = vkb::to_u32(upsample_shader_stages.size());
	upsample_pipeline_create_info.pStages                      = upsample_shader_stages.data();
	upsample_pipeline_create_info.pVertexInputState            = &vertex_input;
	upsample_pipeline_create_info.pInputAssemblyState          = &input_assembly;
	upsample_pipeline_create_info.pRasterizationState          = &raster;
	upsample_pipeline_create_info.pColorBlendState             = &blend;
	upsample_pipeline_create_info.pMultisampleState            = &multisample;
	upsample_pipeline_create_info.pViewportState               = &viewport;
	upsample_pipeline_create_info.pDepthStencilState           = &depth_stencil;
	upsample_pipeline_create_info.pDynamicState                = &dynamic;

	VkGraphicsPipelineCreateInfo save_pipeline_create_info = vkb::initializers::pipeline_create_info(upsample_pipeline_layout, save_render_pass);
	save_pipeline_create_info.stageCount                   = vkb::to_u32(upsample_shader_stages.size());
	save_pipeline_create_info.pStages                      = upsample_shader_stages.data();
	save_pipeline_create_info.pVertexInputState            = &vertex_input;
	save_pipeline_create_info.pInputAssemblyState          = &input_assembly;
	save_pipeline_create_info.pRasterizationState          = &raster;
	save_pipeline_create_info.pColorBlendState             = &blend;
	save_pipeline_create_info.pMultisampleState            = &multisample;
	save_pipeline_create_info.pViewportState               = &viewport;
	save_pipeline_create_info.pDepthStencilState           = &depth_stencil;
	save_pipeline_create_info.pDynamicState                = &dynamic;

	VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), VK_NULL_HANDLE, 1, &upsample_pipeline_create_info, nullptr, &upsample_pipeline));

	VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), VK_NULL_HANDLE, 1, &save_pipeline_create_info, nullptr, &save_pipeline));
}

void color_chart::create_command_pool()
{
	VkCommandPoolCreateInfo command_pool_info = {};
	command_pool_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	command_pool_info.flags                   = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	command_pool_info.queueFamilyIndex        = get_device().get_queue_by_flags(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, 0).get_family_index();
	VK_CHECK(vkCreateCommandPool(get_device().get_handle(), &command_pool_info, nullptr, &cmd_pool));
}

void color_chart::build_command_buffers()
{
	for (int32_t i = 0; i < cmd_buffer_count; ++i)
	{
		recordCommandBuffer(i);
	}
}

void color_chart::rebuild_command_buffers()
{
	for (uint32_t i = 0; i < draw_cmd_buffers.size(); ++i)
	{
		vkWaitForFences(get_device().get_handle(), 1, &wait_fences[i], VK_TRUE, UINT64_MAX);
	}
	vkResetCommandPool(get_device().get_handle(), cmd_pool, 0);
	build_command_buffers();
}

void color_chart::render(float delta_time)
{
	if (!prepared)
	{
		return;
	}
	ApiVulkanSample::prepare_frame();
	vkWaitForFences(get_device().get_handle(), 1, &wait_fences[current_buffer], VK_TRUE, UINT64_MAX);
	vkResetFences(get_device().get_handle(), 1, &wait_fences[current_buffer]);
	vkResetCommandBuffer(draw_cmd_buffers[current_buffer], 0);
	recordCommandBuffer(current_buffer);
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &draw_cmd_buffers[current_buffer];
	VK_CHECK(vkQueueSubmit(queue, 1, &submit_info, wait_fences[current_buffer]));
	ApiVulkanSample::submit_frame();
}

void color_chart::createDescriptorSetLayout()
{
	VkDescriptorSetLayoutBinding    samplerLayoutBinding = vkb::initializers::descriptor_set_layout_binding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 1);
	VkDescriptorSetLayoutCreateInfo layoutInfo           = vkb::initializers::descriptor_set_layout_create_info(&samplerLayoutBinding, 1);
	VK_CHECK(vkCreateDescriptorSetLayout(get_device().get_handle(), &layoutInfo, nullptr, &descriptorSetLayout));
}

void color_chart::createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image, VkDeviceMemory &imageMemory)
{
	VkImageCreateInfo imageInfo = vkb::initializers::image_create_info();
	imageInfo.imageType         = VK_IMAGE_TYPE_2D;
	imageInfo.extent.width      = width;
	imageInfo.extent.height     = height;
	imageInfo.extent.depth      = 1;
	imageInfo.mipLevels         = 1;
	imageInfo.arrayLayers       = 1;
	imageInfo.format            = format;
	imageInfo.tiling            = tiling;
	imageInfo.initialLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.usage             = usage;
	imageInfo.samples           = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.sharingMode       = VK_SHARING_MODE_EXCLUSIVE;

	VK_CHECK(vkCreateImage(get_device().get_handle(), &imageInfo, nullptr, &image));

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(get_device().get_handle(), image, &memRequirements);

	VkMemoryAllocateInfo allocInfo = vkb::initializers::memory_allocate_info();
	allocInfo.allocationSize       = memRequirements.size;
	allocInfo.memoryTypeIndex      = get_device().get_memory_type(memRequirements.memoryTypeBits, properties);

	VK_CHECK(vkAllocateMemory(get_device().get_handle(), &allocInfo, nullptr, &imageMemory));

	vkBindImageMemory(get_device().get_handle(), image, imageMemory, 0);
}

void color_chart::createTextureImageView(const VkImage &textureImage, VkImageView &textureImageView)
{
	VkImageViewCreateInfo viewInfo           = vkb::initializers::image_view_create_info();
	viewInfo.image                           = textureImage;
	viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format                          = SAMPLE_FORMAT;
	viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel   = 0;
	viewInfo.subresourceRange.levelCount     = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount     = 1;

	VK_CHECK(vkCreateImageView(get_device().get_handle(), &viewInfo, nullptr, &textureImageView));
}

void color_chart::createSampleFramebuffer(const VkImageView &textureImageView, VkFramebuffer &framebuffer)
{
	VkFramebufferCreateInfo framebufferInfo = vkb::initializers::framebuffer_create_info();
	framebufferInfo.renderPass              = sample_render_pass;
	framebufferInfo.attachmentCount         = 1;
	framebufferInfo.pAttachments            = &textureImageView;
	framebufferInfo.width                   = SAMPLE_WIDTH;
	framebufferInfo.height                  = SAMPLE_HEIGHT;
	framebufferInfo.layers                  = 1;

	VK_CHECK(vkCreateFramebuffer(get_device().get_handle(), &framebufferInfo, nullptr, &framebuffer));
}

void color_chart::createSavedFramebuffer()
{
	VkFramebufferCreateInfo framebuffer_create_info = {};
	framebuffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	framebuffer_create_info.pNext                   = NULL;
	framebuffer_create_info.renderPass              = save_render_pass;
	framebuffer_create_info.attachmentCount         = 1;
	framebuffer_create_info.pAttachments            = &savedImageView;
	framebuffer_create_info.width                   = SAVE_WIDTH;
	framebuffer_create_info.height                  = SAVE_HEIGHT;
	framebuffer_create_info.layers                  = 1;
	VK_CHECK(vkCreateFramebuffer(device->get_handle(), &framebuffer_create_info, nullptr, &saved_framebuffer));
}

void color_chart::createGeometry()
{
	std::vector<ColoredVertex2D> vertices;
	std::vector<uint16_t>        indices;
	float                        ncols     = (float) COLUMN_COUNT;
	float                        nrows     = (float) ROW_COUNT;
	float                        paddingx  = ((float) PADDING) / SAMPLE_WIDTH;
	float                        paddingy  = paddingx * ncols / nrows;
	float                        spaceSize = 2.0f;
	float                        xSize     = spaceSize - paddingx * 2;
	float                        ySize     = spaceSize - paddingy * 2;
	float                        xoffset   = -xSize / 2;
	float                        yoffset   = -ySize / 2;
	float                        bmax      = nrows * ncols - 1;
	float                        boffset   = 0.0f;
	// float bmax      = 3.0f * nrows * ncols - 1;
	// float boffset = 2.0f * nrows * ncols / bmax;
	int index = vertices.size();
	for (int row = 0; row < ROW_COUNT; ++row)
	{
		for (int col = 0; col < COLUMN_COUNT; ++col)
		{
			float b = ((nrows - 1 - row) * ncols + col) / bmax + boffset;
			// LOGI("row: {}, col: {}, b: {}", row, col, b);
			vertices.emplace_back(ColoredVertex2D{
			    {xSize * col / ncols + xoffset + paddingx, ySize * row / nrows + yoffset + paddingy},
			    {0.0f, 1.0f, b}});
			vertices.emplace_back(ColoredVertex2D{
			    {xSize * (1 + col) / ncols + xoffset - paddingx, ySize * row / nrows + yoffset + paddingy},
			    {1.0f, 1.0f, b}});
			vertices.emplace_back(ColoredVertex2D{
			    {xSize * (1 + col) / ncols + xoffset - paddingx, ySize * (1 + row) / nrows + yoffset - paddingy},
			    {1.0f, 0.0f, b}});
			vertices.emplace_back(ColoredVertex2D{
			    {xSize * col / ncols + xoffset + paddingx, ySize * (1 + row) / nrows + yoffset - paddingy},
			    {0.0f, 0.0f, b}});
			indices.emplace_back(index);
			indices.emplace_back(index + 1);
			indices.emplace_back(index + 2);
			indices.emplace_back(index + 2);
			indices.emplace_back(index + 3);
			indices.emplace_back(index);
			index = vertices.size();
		}
	}

	indexCount = indices.size();
	createVertexBuffer(vertices);
	createIndexBuffer(indices);
}

void color_chart::createVertexBuffer(const std::vector<ColoredVertex2D> &vertices)
{
	VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

	VkBuffer       stagingBuffer;
	VkDeviceMemory stagingBufferMemory;
	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

	void *data;
	vkMapMemory(get_device().get_handle(), stagingBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, vertices.data(), (size_t) bufferSize);
	vkUnmapMemory(get_device().get_handle(), stagingBufferMemory);

	createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

	copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

	vkDestroyBuffer(get_device().get_handle(), stagingBuffer, nullptr);
	vkFreeMemory(get_device().get_handle(), stagingBufferMemory, nullptr);
}

void color_chart::createIndexBuffer(const std::vector<uint16_t> &indices)
{
	VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
	createBuffer(bufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, indexBuffer, indexBufferMemory);

	void *data;
	vkMapMemory(get_device().get_handle(), indexBufferMemory, 0, bufferSize, 0, &data);
	memcpy(data, indices.data(), (size_t) bufferSize);
	vkUnmapMemory(get_device().get_handle(), indexBufferMemory);
}

void color_chart::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory)
{
	VkBufferCreateInfo bufferInfo = vkb::initializers::buffer_create_info(usage, size);
	bufferInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;
	VK_CHECK(vkCreateBuffer(get_device().get_handle(), &bufferInfo, nullptr, &buffer));

	VkMemoryRequirements memRequirements;
	vkGetBufferMemoryRequirements(get_device().get_handle(), buffer, &memRequirements);

	VkMemoryAllocateInfo allocInfo = vkb::initializers::memory_allocate_info();
	allocInfo.allocationSize       = memRequirements.size;
	allocInfo.memoryTypeIndex      = get_device().get_memory_type(memRequirements.memoryTypeBits, properties);
	VK_CHECK(vkAllocateMemory(get_device().get_handle(), &allocInfo, nullptr, &bufferMemory));

	vkBindBufferMemory(get_device().get_handle(), buffer, bufferMemory, 0);
}

void color_chart::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
	VkCommandBuffer commandBuffer = beginSingleTimeCommands();

	VkBufferCopy copyRegion{};
	copyRegion.size = size;
	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

	endSingleTimeCommands(commandBuffer);
}

VkCommandBuffer color_chart::beginSingleTimeCommands()
{
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandPool        = cmd_pool;
	allocInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(get_device().get_handle(), &allocInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	return commandBuffer;
}

void color_chart::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers    = &commandBuffer;

	vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(queue);

	vkFreeCommandBuffers(get_device().get_handle(), cmd_pool, 1, &commandBuffer);
}

void color_chart::createTextureSampler()
{
	VkPhysicalDeviceProperties properties = get_device().get_gpu().get_properties();

	VkSamplerCreateInfo samplerInfo     = vkb::initializers::sampler_create_info();
	samplerInfo.magFilter               = VK_FILTER_NEAREST;
	samplerInfo.minFilter               = VK_FILTER_NEAREST;
	samplerInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable        = VK_FALSE;
	samplerInfo.maxAnisotropy           = properties.limits.maxSamplerAnisotropy;
	samplerInfo.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable           = VK_FALSE;
	samplerInfo.compareOp               = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	samplerInfo.minLod                  = 0.0f;
	samplerInfo.maxLod                  = VK_LOD_CLAMP_NONE;
	samplerInfo.mipLodBias              = 0.0f;

	VK_CHECK(vkCreateSampler(get_device().get_handle(), &samplerInfo, nullptr, &textureSampler));
}

void color_chart::createDescriptorPool()
{
	uint32_t                   count    = static_cast<uint32_t>(cmd_buffer_count);
	VkDescriptorPoolSize       poolSize = vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, count);
	VkDescriptorPoolCreateInfo poolInfo = vkb::initializers::descriptor_pool_create_info(1, &poolSize, count);
	VK_CHECK(vkCreateDescriptorPool(get_device().get_handle(), &poolInfo, nullptr, &descriptor_pool));
}

void color_chart::createDescriptorSets()
{
	std::vector<VkDescriptorSetLayout> layouts(cmd_buffer_count, descriptorSetLayout);
	VkDescriptorSetAllocateInfo        allocInfo = vkb::initializers::descriptor_set_allocate_info(descriptor_pool, layouts.data(), static_cast<uint32_t>(layouts.size()));
	descriptor_sets.resize(cmd_buffer_count);
	VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &allocInfo, descriptor_sets.data()));

	for (uint32_t i = 0; i < cmd_buffer_count; ++i)
	{
		VkDescriptorImageInfo imageInfo       = vkb::initializers::descriptor_image_info(textureSampler, textureImageViews[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		VkWriteDescriptorSet  descriptorWrite = vkb::initializers::write_descriptor_set(descriptor_sets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &imageInfo, 1);
		vkUpdateDescriptorSets(get_device().get_handle(), 1, &descriptorWrite, 0, nullptr);
	}
}

void color_chart::createSaveCommandBuffer()
{
	VkCommandBufferAllocateInfo allocate_info = vkb::initializers::command_buffer_allocate_info(cmd_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
	VK_CHECK(vkAllocateCommandBuffers(get_device().get_handle(), &allocate_info, &saveCommandBuffer));

	VkFenceCreateInfo fenceInfo{};
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
	VK_CHECK(vkCreateFence(get_device().get_handle(), &fenceInfo, nullptr, &savedFence));
}

void color_chart::recordCommandBuffer(uint32_t index)
{
	VkCommandBufferBeginInfo command_buffer_begin_info = vkb::initializers::command_buffer_begin_info();

	// Clear color and depth values.
	VkClearValue clear_values[2];
	clear_values[0].color        = {{0.0f, 0.0f, 0.0f, 1.0f}};
	clear_values[1].depthStencil = {1.0f, 0};

	// Begin the render pass.
	VkRenderPassBeginInfo sample_render_pass_begin_info    = vkb::initializers::render_pass_begin_info();
	sample_render_pass_begin_info.renderPass               = sample_render_pass;
	sample_render_pass_begin_info.renderArea.offset.x      = 0;
	sample_render_pass_begin_info.renderArea.offset.y      = 0;
	sample_render_pass_begin_info.renderArea.extent.width  = SAMPLE_WIDTH;
	sample_render_pass_begin_info.renderArea.extent.height = SAMPLE_HEIGHT;
	sample_render_pass_begin_info.clearValueCount          = 1;
	sample_render_pass_begin_info.pClearValues             = clear_values;

	VkRenderPassBeginInfo render_pass_begin_info    = vkb::initializers::render_pass_begin_info();
	render_pass_begin_info.renderPass               = render_pass;
	render_pass_begin_info.renderArea.offset.x      = 0;
	render_pass_begin_info.renderArea.offset.y      = 0;
	render_pass_begin_info.renderArea.extent.width  = width;
	render_pass_begin_info.renderArea.extent.height = height;
	render_pass_begin_info.clearValueCount          = 2;
	render_pass_begin_info.pClearValues             = clear_values;

	VkRenderPassBeginInfo save_render_pass_begin_info    = vkb::initializers::render_pass_begin_info();
	save_render_pass_begin_info.renderPass               = save_render_pass;
	save_render_pass_begin_info.renderArea.offset.x      = 0;
	save_render_pass_begin_info.renderArea.offset.y      = 0;
	save_render_pass_begin_info.renderArea.extent.width  = SAVE_WIDTH;
	save_render_pass_begin_info.renderArea.extent.height = SAVE_HEIGHT;
	save_render_pass_begin_info.clearValueCount          = 1;
	save_render_pass_begin_info.pClearValues             = clear_values;
	save_render_pass_begin_info.framebuffer              = saved_framebuffer;

	auto cmd = index == draw_cmd_buffers.size() ? saveCommandBuffer : draw_cmd_buffers[index];

	// Begin command buffer.
	vkBeginCommandBuffer(cmd, &command_buffer_begin_info);

	// Set framebuffer for this command buffer.
	sample_render_pass_begin_info.framebuffer = sample_framebuffers[index];

	// We will add draw commands in the same command buffer.
	vkCmdBeginRenderPass(cmd, &sample_render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

	// Bind the graphics pipeline.
	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sample_pipeline);

	// Set viewport dynamically
	VkViewport viewport = vkb::initializers::viewport(static_cast<float>(SAMPLE_WIDTH), static_cast<float>(SAMPLE_HEIGHT), 0.0f, 1.0f);
	vkCmdSetViewport(cmd, 0, 1, &viewport);

	// Set scissor dynamically
	VkRect2D scissor = vkb::initializers::rect2D(SAMPLE_WIDTH, SAMPLE_HEIGHT, 0, 0);
	vkCmdSetScissor(cmd, 0, 1, &scissor);

	// Bind push constants
	auto         currentTime = std::chrono::high_resolution_clock::now();
	float        time        = std::chrono::duration_cast<std::chrono::duration<float>>(currentTime - startTime).count();
	float        dt          = std::chrono::duration_cast<std::chrono::duration<float>>(currentTime - previousTime).count();
	PushConstant pc{time, dt};
	vkCmdPushConstants(cmd, sample_pipeline_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstant), &pc);
	if (time > MAX_TIME)
	{
		startTime = currentTime;
	}
	previousTime = currentTime;

	// Bind geometry
	VkBuffer     vertexBuffers[] = {vertexBuffer};
	VkDeviceSize offsets[]       = {0};
	vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
	vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

	// Draw
	vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);

	// Complete render pass.
	vkCmdEndRenderPass(cmd);

	// Upsample render pass
	// Set framebuffer for this command buffer.
	if (index == draw_cmd_buffers.size())
	{
		// Save pass
		vkCmdBeginRenderPass(cmd, &save_render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, save_pipeline);
		VkViewport save_viewport = vkb::initializers::viewport(static_cast<float>(SAVE_WIDTH), static_cast<float>(SAVE_HEIGHT), 0.0f, 1.0f);
		vkCmdSetViewport(cmd, 0, 1, &save_viewport);
		VkRect2D save_scissor = vkb::initializers::rect2D(SAVE_WIDTH, SAVE_HEIGHT, 0, 0);
		vkCmdSetScissor(cmd, 0, 1, &save_scissor);
	}
	else
	{
		// Presentation pass
		render_pass_begin_info.framebuffer = framebuffers[index];
		vkCmdBeginRenderPass(cmd, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, upsample_pipeline);
		VkViewport upsample_viewport = vkb::initializers::viewport(static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
		vkCmdSetViewport(cmd, 0, 1, &upsample_viewport);
		VkRect2D upsample_scissor = vkb::initializers::rect2D(width, height, 0, 0);
		vkCmdSetScissor(cmd, 0, 1, &upsample_scissor);
	}

	// Bind descriptor
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, upsample_pipeline_layout, 0, 1, &descriptor_sets[index], 0, nullptr);

	// Draw
	vkCmdDraw(cmd, 6, 1, 0, 0);

	// Draw user interface.
	if (DRAW_UI)
		draw_ui(cmd);

	// Complete render pass.
	vkCmdEndRenderPass(cmd);

	// Complete the command buffer.
	VK_CHECK(vkEndCommandBuffer(cmd));
}

void color_chart::exportImage()
{
	vkWaitForFences(get_device().get_handle(), 1, &savedFence, VK_TRUE, UINT64_MAX);
	vkResetFences(get_device().get_handle(), 1, &savedFence);
	LOGI("Exporting image...");
	VkSubmitInfo submitInfo{};
	submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers    = &saveCommandBuffer;
	VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, savedFence));

	vkWaitForFences(get_device().get_handle(), 1, &savedFence, VK_TRUE, UINT64_MAX);

	// Get layout of the image (including row pitch)
	VkImageSubresource  subResource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
	VkSubresourceLayout subResourceLayout;
	vkGetImageSubresourceLayout(get_device().get_handle(), savedImage, &subResource, &subResourceLayout);

	// Map image memory so we can start copying from it
	uint8_t *raw_data;
	vkMapMemory(get_device().get_handle(), savedImageMemory, 0, VK_WHOLE_SIZE, 0, (void **) &raw_data);
	raw_data += subResourceLayout.offset;

	// vkb::fs::write_image_hdr(raw_data, SAVED_IMAGE_FILENAME, SAVE_WIDTH, SAVE_HEIGHT, SAVE_COMPONENTS);
	vkb::fs::write_image(raw_data, SAVED_IMAGE_FILENAME, SAVE_WIDTH, SAVE_HEIGHT, SAVE_COMPONENTS, SAVE_WIDTH * SAVE_COMPONENTS);

	LOGI("Image saved to disk {}{}.png", vkb::fs::path::relative_paths.find(vkb::fs::path::Type::Screenshots)->second, SAVED_IMAGE_FILENAME);

	// Clean up resources
	vkUnmapMemory(get_device().get_handle(), savedImageMemory);
}

std::unique_ptr<vkb::VulkanSample> create_color_chart()
{
	return std::make_unique<color_chart>();
}
