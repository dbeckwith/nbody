#version 450 core

uniform mat4 view; // camera view matrix
uniform mat4 proj; // camera projection matrix

in vec2 sprite_vertex; // sprite vertex position
in vec2 sprite_uv; // sprite texture coordinate

in float particle_radius; // particle radius
in float particle_mass; // particle mass
in vec3 particle_position; // particle position
in vec3 particle_velocity; // particle velocity

out vec2 frag_uv; // pass UV to fragment shader
out float frag_mass; // pass mass to fragment shader
out vec3 frag_velocity; // pass velocity to fragment shader

void main() {
    frag_uv = sprite_uv;
    frag_mass = particle_mass;
    frag_velocity = particle_velocity;

    // get camera right and up vector
    vec3 cam_right = vec3(view[0][0], view[1][0], view[2][0]);
    vec3 cam_up    = vec3(view[0][1], view[1][1], view[2][1]);

    // project vertex in camera space
    // will always render sprites facing camera ("billboard")
    vec3 vert = (sprite_vertex.x * cam_right + sprite_vertex.y * cam_up);
    vert *= particle_radius * 1.0;

    // project vertex into view space
    gl_Position = proj * view * vec4(vert + particle_position, 1.0);
}
