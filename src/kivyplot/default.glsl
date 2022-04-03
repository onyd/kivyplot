---VERTEX SHADER-------------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

// inside VS, read-only
attribute vec3 v_pos;
attribute vec3 v_normal;

// from python, read-only
uniform mat4 modelview_mat;
uniform mat4 projection_mat;
uniform mat4 normal_mat;
uniform mat4 model_mat;
uniform mat4 view_mat;

// used later in FS
varying vec3 normal_vec;


void main (void) {
    // fetch read-only for later use
    normal_vec = v_normal;

    vec4 pos = modelview_mat * vec4(v_pos, 1.0);

    // required shader clip-space output
    gl_Position = projection_mat * pos;
}


---FRAGMENT SHADER-----------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

varying vec3 normal_vec;

uniform mat4 modelview_mat;
uniform vec3 Ka; // color (ambient)
uniform float Tr; // transparency

uniform vec3 camera_pos;
uniform vec3 camera_back;
uniform mat4 normal_mat;
uniform float t;

void main (void) {
    vec4 v_normal = normalize(normal_mat * vec4(normal_vec, 0));

    gl_FragColor = vec4(Ka *t, Tr);//* max(dot(v_normal, vec4(camera_back, 1)), 0), Tr);
}
