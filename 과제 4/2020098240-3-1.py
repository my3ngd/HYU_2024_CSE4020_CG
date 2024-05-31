import glfw
from OpenGL.GL import * 
from OpenGL.GLU import *
import numpy as np
import pdb

key_value = 0
T = np.identity(3)  # actually it is not necessary to use global variable..

def keyboard_callback(wind, key, scancode, action, mods):
    global key_value
    if action == glfw.PRESS:
        if key == glfw.KEY_Q: key_value = 1
        if key == glfw.KEY_E: key_value = 2
        if key == glfw.KEY_A: key_value = 3
        if key == glfw.KEY_D: key_value = 4
        if key == glfw.KEY_1: key_value = -1
    return

def render(T):
    glClear(GL_COLOR_BUFFER_BIT)
    glLoadIdentity()
    # draw cooridnate
    glBegin(GL_LINES)
    glColor3ub(255, 0, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([1.,0.]))
    glColor3ub(0, 255, 0)
    glVertex2fv(np.array([0.,0.]))
    glVertex2fv(np.array([0.,1.]))
    glEnd()
    # draw triangle
    glBegin(GL_TRIANGLES)
    glColor3ub(255, 255, 255)
    glVertex2fv( (T @ np.array([.0,.5,1.]))[:-1] )
    glVertex2fv( (T @ np.array([.0,.0,1.]))[:-1] )
    glVertex2fv( (T @ np.array([.5,.0,1.]))[:-1] )
    glEnd()

def calc():
    theta = np.pi/18
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta),  np.cos(theta), 0],
                    [            0,              0, 1]])
    global key_value, T
    if key_value == 0:  # none
        return
    if key_value == 1:  # Q
        key_value = 0
        T[0:1, 2] -= 0.1
        return
    if key_value == 2:  # E
        key_value = 0
        T[0:1, 2] += 0.1
        return
    if key_value == 3:  # A (rotate CCW 10deg)
        key_value = 0
        T = T @ R    # right-multiplication by R
    if key_value == 4:
        key_value = 0
        T = T @ np.linalg.inv(R)
    if key_value == -1:  # D (rotate CW 10deg)
        key_value = 0
        T = np.identity(3)  # right-multiplication by R

def main():
    global key_value, T
    if not glfw.init():
        return
    window = glfw.create_window(480, 480, "2020098240-3-1", None, None)
    if not window:
        glfw.terminate()
        return

    # settings
    glfw.set_key_callback(window=window, cbfun=keyboard_callback)
    glfw.make_context_current(window=window)

    # main loop
    while not glfw.window_should_close(window=window):
        glfw.poll_events()
        calc()
        render(T)
        glfw.swap_buffers(window=window)

    glfw.terminate()
    return


if __name__ == "__main__":
    main()
