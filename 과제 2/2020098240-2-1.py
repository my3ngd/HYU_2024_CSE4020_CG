import glfw, numpy as np
from OpenGL.GL import *

key_value = GL_LINE_LOOP

def keyboard_callback(window, key, scancode, action, mods):
    global key_value
    if action == glfw.PRESS:
        if key == glfw.KEY_1:  key_value = GL_POINTS
        if key == glfw.KEY_2:  key_value = GL_LINES
        if key == glfw.KEY_3:  key_value = GL_LINE_STRIP
        if key == glfw.KEY_4:  key_value = GL_LINE_LOOP
        if key == glfw.KEY_5:  key_value = GL_TRIANGLES
        if key == glfw.KEY_6:  key_value = GL_TRIANGLE_STRIP
        if key == glfw.KEY_7:  key_value = GL_TRIANGLE_FAN
        if key == glfw.KEY_8:  key_value = GL_QUADS
        if key == glfw.KEY_9:  key_value = GL_QUAD_STRIP
        if key == glfw.KEY_0:  key_value = GL_POLYGON
    return
    # verbose
    print(key_value)

def render():
    key = key_value
    glClear(GL_COLOR_BUFFER_BIT)
    num_vertex = 12
    angles = np.linspace(0, np.pi*2, num_vertex+1)
    x_coord = np.cos(angles)
    y_coord = np.sin(angles)
    vertexes = np.stack((x_coord, y_coord), axis=1)
    glBegin(key)
    for i in range(num_vertex):
        glVertex(vertexes[i])
    glEnd()


def main():
    global key_value
    if not glfw.init():
        return
    window = glfw.create_window(480, 480, "2020098240-2-1", None, None)
    if not window:
        glfw.terminate()
        return

    # settings
    glfw.set_key_callback(window=window, cbfun=keyboard_callback)
    glfw.make_context_current(window=window)

    # main render loop
    while not glfw.window_should_close(window=window):
        glfw.poll_events()
        render()
        glfw.swap_buffers(window=window)
    glfw.terminate()
    return

if __name__ == "__main__":
    main()
