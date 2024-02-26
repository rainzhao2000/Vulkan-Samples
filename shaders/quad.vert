#version 450

layout(location = 0) out vec2 out_texcoord;

vec2 quad_positions[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 1.0),
    vec2(1.0, 1.0),
    vec2(-1.0, 1.0),
    vec2(-1.0, -1.0)
);

vec2 texture_coords[6] = vec2[](
    vec2(0.0, 0.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0),
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(0.0, 0.0)
);

void main()
{
    gl_Position = vec4(quad_positions[gl_VertexIndex], 0.5, 1.0);

    out_texcoord = texture_coords[gl_VertexIndex];
}
