#version 450

layout(push_constant, std430) uniform pc {
    float time; // seconds
    float dt; // seconds
};

layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

const float PI = 3.141592;

void main() {
    //outColor = vec4(sin(1.0 * PI * (fragColor.r + time)) / 2.0 + 0.5, fragColor.gb, 1.0);
    outColor = vec4(fragColor, 1.0);
}