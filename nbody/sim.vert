#version 450 core

uniform mat4 mvp;

in vec2 sprite_vertex;
in vec2 sprite_uv;

in float particle_radius;
in vec3 particle_position;

out vec2 uv;

void main() {
    uv = sprite_uv;

    gl_Position = mvp * vec4(vec3(sprite_vertex, 0) * particle_radius + particle_position, 1.0);
    gl_Position.w *= -1;
    // gl_Position /= gl_Position.w;
}
