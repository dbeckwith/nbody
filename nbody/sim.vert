#version 330

uniform mat4 mvp;

in vec3 position;

void main() {
    vec4 p = mvp * vec4(position, 1.0);
    gl_Position = p / p.w;
}
