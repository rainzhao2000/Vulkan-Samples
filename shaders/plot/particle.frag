#version 450
/* Copyright (c) 2019, Sascha Willems
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

layout (binding = 0) uniform sampler2D samplerColorMap;
layout (binding = 1) uniform sampler2D samplerGradientRamp;

//layout (location = 0) in float inGradientPos;
layout (location = 0) in vec3 inPos;

layout (location = 0) out vec4 outFragColor;

mat3 LMS_to_XYZ = mat3(
	1.91020, -1.11212, 0.20191,
	0.37095, 0.62905, 0.0,
	0.0, 0.0, 1.0
);
mat3 LMSD65_to_XYZD65 = mat3(
	1.86007, -1.12948, 0.219898,
	0.361223, 0.638804, -0.000007,
	0.0, 0.0, 1.08909
);
mat3 XYZD65_to_RGB = mat3(
	3.2406, -1.5372, -0.4986,
	-0.9689, 1.8758, 0.0415,
	0.0557, -0.2040, 1.0570
);

void main () 
{
	outFragColor.rgb = XYZD65_to_RGB * inPos;
}
