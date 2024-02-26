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

const uint32_t SAMPLE_WIDTH  = 256;
const uint32_t SAMPLE_HEIGHT = 128;
const VkFormat SAMPLE_FORMAT = VK_FORMAT_R16G16B16A16_UNORM;

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
		vkDestroyPipeline(get_device().get_handle(), upsample_pipeline, nullptr);
		vkDestroyPipelineLayout(get_device().get_handle(), upsample_pipeline_layout, nullptr);

		vkDestroyPipeline(get_device().get_handle(), sample_pipeline, nullptr);
		vkDestroyPipelineLayout(get_device().get_handle(), sample_pipeline_layout, nullptr);

		vkDestroyDescriptorPool(get_device().get_handle(), descriptor_pool, nullptr);

		vkDestroySampler(get_device().get_handle(), textureSampler, nullptr);
		vkDestroyImageView(get_device().get_handle(), textureImageView, nullptr);
		vkDestroyImage(get_device().get_handle(), textureImage, nullptr);
		vkFreeMemory(get_device().get_handle(), textureImageMemory, nullptr);

		vkDestroyDescriptorSetLayout(get_device().get_handle(), descriptorSetLayout, nullptr);

		vkDestroyBuffer(get_device().get_handle(), indexBuffer, nullptr);
		vkFreeMemory(get_device().get_handle(), indexBufferMemory, nullptr);

		vkDestroyBuffer(get_device().get_handle(), vertexBuffer, nullptr);
		vkFreeMemory(get_device().get_handle(), vertexBufferMemory, nullptr);
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
	createTextureSampler();
	createGeometry();
	createDescriptorPool();
	createDescriptorSets();
	build_command_buffers();
	prepared = true;
	return true;
}

void color_chart::create_render_context()
{
	// Use wide color gamut hdr when possible
	auto surface_priority_list = std::vector<VkSurfaceFormatKHR>{{VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT},
	                                                             {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
	                                                             {VK_FORMAT_R8G8B8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR}};

	VulkanSample::create_render_context(surface_priority_list);

	createImage(
	    SAMPLE_WIDTH,
	    SAMPLE_HEIGHT,
	    SAMPLE_FORMAT,
	    VK_IMAGE_TILING_OPTIMAL,
	    VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
	    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	    textureImage,
	    textureImageMemory);
	createTextureImageView();
}

void color_chart::setup_render_pass()
{
	std::array<VkAttachmentDescription, 2> attachments = {};
	// Color attachment for first subpass
	attachments[0].format         = SAMPLE_FORMAT;
	attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
	attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[0].finalLayout    = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	// Color attachment for second subpass
	attachments[1].format         = render_context->get_format();
	attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
	attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
	attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	attachments[1].finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference color_reference0 = {};
	color_reference0.attachment            = 0;
	color_reference0.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentReference color_reference1 = {};
	color_reference1.attachment            = 1;
	color_reference1.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	std::array<VkSubpassDescription, 2> subpass_descriptions = {};
	// subpass0
	subpass_descriptions[0].pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass_descriptions[0].colorAttachmentCount    = 1;
	subpass_descriptions[0].pColorAttachments       = &color_reference0;
	subpass_descriptions[0].inputAttachmentCount    = 0;
	subpass_descriptions[0].pInputAttachments       = nullptr;
	subpass_descriptions[0].preserveAttachmentCount = 0;
	subpass_descriptions[0].pPreserveAttachments    = nullptr;
	subpass_descriptions[0].pResolveAttachments     = nullptr;
	// subpass1
	subpass_descriptions[1].pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass_descriptions[1].colorAttachmentCount    = 1;
	subpass_descriptions[1].pColorAttachments       = &color_reference1;
	subpass_descriptions[1].inputAttachmentCount    = 0;
	subpass_descriptions[1].pInputAttachments       = nullptr;
	subpass_descriptions[1].preserveAttachmentCount = 0;
	subpass_descriptions[1].pPreserveAttachments    = nullptr;
	subpass_descriptions[1].pResolveAttachments     = nullptr;

	// Subpass dependencies for layout transitions
	std::array<VkSubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass      = 0;
	dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[0].srcAccessMask   = VK_ACCESS_NONE_KHR;
	dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	dependencies[1].srcSubpass      = 0;
	dependencies[1].dstSubpass      = 1;
	dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
	dependencies[1].srcAccessMask   = VK_ACCESS_NONE_KHR;
	dependencies[1].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
	dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

	VkRenderPassCreateInfo render_pass_create_info = {};
	render_pass_create_info.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
	render_pass_create_info.attachmentCount        = static_cast<uint32_t>(attachments.size());
	render_pass_create_info.pAttachments           = attachments.data();
	render_pass_create_info.subpassCount           = static_cast<uint32_t>(subpass_descriptions.size());
	render_pass_create_info.pSubpasses             = subpass_descriptions.data();
	render_pass_create_info.dependencyCount        = static_cast<uint32_t>(dependencies.size());
	render_pass_create_info.pDependencies          = dependencies.data();

	VK_CHECK(vkCreateRenderPass(get_device().get_handle(), &render_pass_create_info, nullptr, &render_pass));
}

void color_chart::setup_framebuffer()
{
	// Delete existing frame buffers
	if (framebuffers.size() > 0)
	{
		for (uint32_t i = 0; i < framebuffers.size(); i++)
		{
			if (framebuffers[i] != VK_NULL_HANDLE)
			{
				vkDestroyFramebuffer(device->get_handle(), framebuffers[i], nullptr);
			}
		}
	}

	// Create frame buffers for every swap chain image
	framebuffers.resize(render_context->get_render_frames().size());
	for (uint32_t i = 0; i < framebuffers.size(); i++)
	{
		std::array<VkImageView, 2> attachments = {
		    textureImageView,
		    swapchain_buffers[i].view};

		VkFramebufferCreateInfo framebuffer_create_info = {};
		framebuffer_create_info.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		framebuffer_create_info.pNext                   = NULL;
		framebuffer_create_info.renderPass              = render_pass;
		framebuffer_create_info.attachmentCount         = 2;
		framebuffer_create_info.pAttachments            = attachments.data();
		framebuffer_create_info.width                   = SAMPLE_WIDTH;
		framebuffer_create_info.height                  = SAMPLE_HEIGHT;
		framebuffer_create_info.layers                  = 1;

		VK_CHECK(vkCreateFramebuffer(device->get_handle(), &framebuffer_create_info, nullptr, &framebuffers[i]));
	}
}

void color_chart::prepare_pipelines()
{
	// Create a blank pipeline layout.
	// We are not binding any resources to the pipeline in this sample.
	VkPipelineLayoutCreateInfo layout_info = vkb::initializers::pipeline_layout_create_info(nullptr, 0);
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

	// Disable depth testing
	VkPipelineDepthStencilStateCreateInfo depth_stencil = vkb::initializers::pipeline_depth_stencil_state_create_info(VK_FALSE, VK_FALSE, VK_COMPARE_OP_GREATER);

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
	VkGraphicsPipelineCreateInfo pipeline_create_info = vkb::initializers::pipeline_create_info(sample_pipeline_layout, render_pass);
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
	pipeline_create_info.subpass                      = 0;

	VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1, &pipeline_create_info, nullptr, &sample_pipeline));

	// Upsampling pipeline
	layout_info = vkb::initializers::pipeline_layout_create_info(&descriptorSetLayout, 1);
	VK_CHECK(vkCreatePipelineLayout(get_device().get_handle(), &layout_info, nullptr, &upsample_pipeline_layout));

	shader_stages[0] = load_shader("quad.vert", VK_SHADER_STAGE_VERTEX_BIT);
	shader_stages[1] = load_shader("texture.frag", VK_SHADER_STAGE_FRAGMENT_BIT);

	vertex_input = vkb::initializers::pipeline_vertex_input_state_create_info();

	pipeline_create_info                     = vkb::initializers::pipeline_create_info(upsample_pipeline_layout, render_pass);
	pipeline_create_info.stageCount          = vkb::to_u32(shader_stages.size());
	pipeline_create_info.pStages             = shader_stages.data();
	pipeline_create_info.pVertexInputState   = &vertex_input;
	pipeline_create_info.pInputAssemblyState = &input_assembly;
	pipeline_create_info.pRasterizationState = &raster;
	pipeline_create_info.pColorBlendState    = &blend;
	pipeline_create_info.pMultisampleState   = &multisample;
	pipeline_create_info.pViewportState      = &viewport;
	pipeline_create_info.pDepthStencilState  = &depth_stencil;
	pipeline_create_info.pDynamicState       = &dynamic;
	pipeline_create_info.subpass             = 1;

	VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1, &pipeline_create_info, nullptr, &upsample_pipeline));
}

void color_chart::build_command_buffers()
{
	VkCommandBufferBeginInfo command_buffer_begin_info = vkb::initializers::command_buffer_begin_info();

	// Clear color and depth values.
	VkClearValue clear_values[2];
	clear_values[0].color        = {{0.0f, 0.0f, 0.0f, 0.0f}};
	clear_values[1].depthStencil = {0.0f, 0};

	// Begin the render pass.
	VkRenderPassBeginInfo render_pass_begin_info    = vkb::initializers::render_pass_begin_info();
	render_pass_begin_info.renderPass               = render_pass;
	render_pass_begin_info.renderArea.offset.x      = 0;
	render_pass_begin_info.renderArea.offset.y      = 0;
	render_pass_begin_info.renderArea.extent.width  = width;
	render_pass_begin_info.renderArea.extent.height = height;
	render_pass_begin_info.clearValueCount          = 2;
	render_pass_begin_info.pClearValues             = clear_values;

	for (int32_t i = 0; i < draw_cmd_buffers.size(); ++i)
	{
		auto cmd = draw_cmd_buffers[i];

		// Begin command buffer.
		vkBeginCommandBuffer(cmd, &command_buffer_begin_info);

		// Set framebuffer for this command buffer.
		render_pass_begin_info.framebuffer = framebuffers[i];
		// We will add draw commands in the same command buffer.
		vkCmdBeginRenderPass(cmd, &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);

		// Bind the graphics pipeline.
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sample_pipeline);

		// Set viewport dynamically
		VkViewport viewport = vkb::initializers::viewport(static_cast<float>(SAMPLE_WIDTH), static_cast<float>(SAMPLE_HEIGHT), 0.0f, 1.0f);
		vkCmdSetViewport(cmd, 0, 1, &viewport);

		// Set scissor dynamically
		VkRect2D scissor = vkb::initializers::rect2D(SAMPLE_WIDTH, SAMPLE_HEIGHT, 0, 0);
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		// Bind geometry
		VkBuffer     vertexBuffers[] = {vertexBuffer};
		VkDeviceSize offsets[]       = {0};
		vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

		// Draw
		vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);

		vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);

		// Bind the graphics pipeline.
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, upsample_pipeline);

		// Set viewport dynamically
		viewport = vkb::initializers::viewport(static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
		vkCmdSetViewport(cmd, 0, 1, &viewport);

		// Set scissor dynamically
		scissor = vkb::initializers::rect2D(width, height, 0, 0);
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		// Bind descriptor sets
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, upsample_pipeline_layout, 0, 1, &descriptorSets[i], 0, nullptr);

		// Draw
		vkCmdDraw(cmd, 6, 1, 0, 0);

		// Draw user interface.
		draw_ui(cmd);

		// Complete render pass.
		vkCmdEndRenderPass(cmd);

		// Complete the command buffer.
		VK_CHECK(vkEndCommandBuffer(cmd));
	}
}

void color_chart::render(float delta_time)
{
	if (!prepared)
	{
		return;
	}
	ApiVulkanSample::prepare_frame();
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers    = &draw_cmd_buffers[current_buffer];
	VK_CHECK(vkQueueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE));
	ApiVulkanSample::submit_frame();
}

void color_chart::createGeometry()
{
	std::vector<ColoredVertex2D> vertices;
	std::vector<uint16_t>        indices;
	int                          nrows = 4;
	int                          ncols = 8;
	// int   nrows     = 3;
	// int   ncols     = 7;
	float paddingx  = 0.0f;        // 0.005f;
	float paddingy  = 0.0f;        // 0.01f;
	float spaceSize = 2.0f;
	float xSize     = spaceSize - paddingx * 2;
	float ySize     = spaceSize - paddingy * 2;
	float xoffset   = -xSize / 2;
	float yoffset   = -ySize / 2;
	float bmax      = nrows * ncols - 1;
	float boffset   = 0.0f;
	// float bmax      = 3.0f * nrows * ncols - 1;
	// float boffset = 2.0f * nrows * ncols / bmax;
	int index = vertices.size();
	for (int row = 0; row < nrows; ++row)
	{
		for (int col = 0; col < ncols; ++col)
		{
			float b = ((nrows - 1 - row) * ncols + col) / bmax + boffset;
			// LOGI("row: {}, col: {}, b: {}", row, col, b);
			vertices.emplace_back(ColoredVertex2D{{xSize * col / ncols + xoffset + paddingx, ySize * row / nrows + yoffset + paddingy}, {0.0f, 1.0f, b}});
			vertices.emplace_back(ColoredVertex2D{{xSize * (1 + col) / ncols + xoffset - paddingx, ySize * row / nrows + yoffset + paddingy}, {1.0f, 1.0f, b}});
			vertices.emplace_back(ColoredVertex2D{{xSize * (1 + col) / ncols + xoffset - paddingx, ySize * (1 + row) / nrows + yoffset - paddingy}, {1.0f, 0.0f, b}});
			vertices.emplace_back(ColoredVertex2D{{xSize * col / ncols + xoffset + paddingx, ySize * (1 + row) / nrows + yoffset - paddingy}, {0.0f, 0.0f, b}});
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

void color_chart::createTextureImageView()
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

void color_chart::createTextureSampler()
{
	VkPhysicalDeviceProperties properties = get_device().get_gpu().get_properties();

	VkSamplerCreateInfo samplerInfo     = vkb::initializers::sampler_create_info();
	samplerInfo.magFilter               = VK_FILTER_NEAREST;
	samplerInfo.minFilter               = VK_FILTER_NEAREST;
	samplerInfo.addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable        = VK_TRUE;
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
	uint32_t                   cmdBufferCount = static_cast<uint32_t>(draw_cmd_buffers.size());
	VkDescriptorPoolSize       poolSize       = vkb::initializers::descriptor_pool_size(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, cmdBufferCount);
	VkDescriptorPoolCreateInfo poolInfo       = vkb::initializers::descriptor_pool_create_info(1, &poolSize, cmdBufferCount);
	VK_CHECK(vkCreateDescriptorPool(get_device().get_handle(), &poolInfo, nullptr, &descriptor_pool));
}

void color_chart::createDescriptorSets()
{
	uint32_t                           cmdBufferCount = static_cast<uint32_t>(draw_cmd_buffers.size());
	std::vector<VkDescriptorSetLayout> layouts(cmdBufferCount, descriptorSetLayout);
	VkDescriptorSetAllocateInfo        allocInfo = vkb::initializers::descriptor_set_allocate_info(descriptor_pool, layouts.data(), cmdBufferCount);

	descriptorSets.resize(cmdBufferCount);
	VK_CHECK(vkAllocateDescriptorSets(get_device().get_handle(), &allocInfo, descriptorSets.data()));

	for (uint32_t i = 0; i < cmdBufferCount; i++)
	{
		VkDescriptorImageInfo imageInfo       = vkb::initializers::descriptor_image_info(textureSampler, textureImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
		VkWriteDescriptorSet  descriptorWrite = vkb::initializers::write_descriptor_set(descriptorSets[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &imageInfo, 1);
		vkUpdateDescriptorSets(get_device().get_handle(), 1, &descriptorWrite, 0, nullptr);
	}
}

std::unique_ptr<vkb::VulkanSample> create_color_chart()
{
	return std::make_unique<color_chart>();
}
