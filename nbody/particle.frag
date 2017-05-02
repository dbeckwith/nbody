#version 450 core

uniform float collision_overlap;
uniform uint color_mode;

in vec2 frag_uv;
in float frag_mass;
in vec3 frag_velocity;

out vec4 color;

vec3 hsv2rgb(vec3 c) {
    // http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    // generate radial alpha gradient to look like a sphere, based on collision overlap
    float r = length(frag_uv - 0.5) * 2.0;
    float alpha = 1.0 - smoothstep(1.0 - sqrt(1.0 - (1.0 - collision_overlap) * (1.0 - collision_overlap)), 1.0, r);
    // float alpha = r > 1.0 ? 0.0 : 1.0; // use with multisampling
    float hue = 0.0;
    if (color_mode == 0) {
        // hue based on mass
        hue = log(frag_mass) / log(15.0);
    }
    else if (color_mode == 1) {
        // hue based on velocity
        hue = length(frag_velocity) * 0.05;
    }
    color = vec4(hsv2rgb(vec3(hue, 1.0, 1.0)), alpha);
}
