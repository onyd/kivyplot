---VERTEX SHADER-------------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

// inside VS, read-only
attribute vec3 v_pos;

// from python, read-only
uniform mat4 modelview_mat;
uniform mat4 projection_mat;
uniform mat4 model_mat;
uniform mat4 view_mat;

void main (void) {
    vec4 pos = modelview_mat * vec4(v_pos, 1.0);
    // required shader clip-space output
    gl_Position = projection_mat * pos;
}


---FRAGMENT SHADER-----------------------------------------------------
#ifdef GL_ES
    precision highp float;
#endif

uniform vec3 Ka; // color (ambient)
uniform float Tr; // transparency

void main (void){
    gl_FragColor = vec4(Ka, Tr);
}
