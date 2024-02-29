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

#pragma once

#include "api_vulkan_sample.h"

struct ColoredVertex2D
{
	glm::vec2 pos;
	glm::vec3 color;

	static VkVertexInputBindingDescription                  getBindingDescription();
	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions();
};

class color_chart : public ApiVulkanSample
{
  public:
	color_chart();
	virtual ~color_chart();

	// Create pipeline
	void prepare_pipelines();

	// Override basic framework functionality
	void build_command_buffers() override;
	void render(float delta_time) override;
	bool prepare(const vkb::ApplicationOptions &options) override;
	void create_render_context() override;
	void setup_render_pass() override;

  private:
	// Sample specific data
	std::vector<VkImage>         textureImages;
	std::vector<VkDeviceMemory>  textureImageMemories;
	std::vector<VkImageView>     textureImageViews;
	VkRenderPass                 sample_render_pass;
	VkDescriptorSetLayout        descriptorSetLayout;
	VkPipeline                   sample_pipeline{};
	VkPipelineLayout             sample_pipeline_layout{};
	VkPipeline                   upsample_pipeline{};
	VkPipelineLayout             upsample_pipeline_layout{};
	std::vector<VkFramebuffer>   sample_framebuffers;
	VkBuffer                     vertexBuffer;
	VkDeviceMemory               vertexBufferMemory;
	VkBuffer                     indexBuffer;
	VkDeviceMemory               indexBufferMemory;
	uint32_t                     indexCount;
	VkSampler                    textureSampler;
	std::vector<VkDescriptorSet> descriptor_sets;

	void            createDescriptorSetLayout();
	void            createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image, VkDeviceMemory &imageMemory);
	void            createTextureImageView(const VkImage &textureImage, VkImageView &textureImageView);
	void            createSampleFramebuffer(const VkImageView &textureImageView, VkFramebuffer &framebuffer);
	void            createGeometry();
	void            createVertexBuffer(const std::vector<ColoredVertex2D> &vertices);
	void            createIndexBuffer(const std::vector<uint16_t> &indices);
	void            createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory);
	void            copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
	VkCommandBuffer beginSingleTimeCommands();
	void            endSingleTimeCommands(VkCommandBuffer commandBuffer);
	void            createTextureSampler();
	void            createDescriptorPool();
	void            createDescriptorSets();
};

std::unique_ptr<vkb::VulkanSample> create_color_chart();
