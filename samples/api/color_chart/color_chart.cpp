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
		vkDestroyBuffer(get_device().get_handle(), indexBuffer, nullptr);
		vkFreeMemory(get_device().get_handle(), indexBufferMemory, nullptr);
		vkDestroyBuffer(get_device().get_handle(), vertexBuffer, nullptr);
		vkFreeMemory(get_device().get_handle(), vertexBufferMemory, nullptr);
		vkDestroyPipeline(get_device().get_handle(), sample_pipeline, nullptr);
		vkDestroyPipelineLayout(get_device().get_handle(), sample_pipeline_layout, nullptr);
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

	prepare_pipelines();
	createGeometry();
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

	// Enable depth testing (using reversed depth-buffer for increased precision).
	VkPipelineDepthStencilStateCreateInfo depth_stencil = vkb::initializers::pipeline_depth_stencil_state_create_info(VK_TRUE, VK_TRUE, VK_COMPARE_OP_GREATER);

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

	VK_CHECK(vkCreateGraphicsPipelines(get_device().get_handle(), pipeline_cache, 1, &pipeline_create_info, nullptr, &sample_pipeline));
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
		VkViewport viewport = vkb::initializers::viewport(static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f);
		vkCmdSetViewport(cmd, 0, 1, &viewport);

		// Set scissor dynamically
		VkRect2D scissor = vkb::initializers::rect2D(width, height, 0, 0);
		vkCmdSetScissor(cmd, 0, 1, &scissor);

		// Bind geometry
		VkBuffer     vertexBuffers[] = {vertexBuffer};
		VkDeviceSize offsets[]       = {0};
		vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
		vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

		// Draw
		vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);

		// Draw user interface.
		draw_ui(draw_cmd_buffers[i]);

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

std::unique_ptr<vkb::VulkanSample> create_color_chart()
{
	return std::make_unique<color_chart>();
}
