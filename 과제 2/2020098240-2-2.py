import glfw, numpy as np
from OpenGL.GL import *


# render function is given.
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


def main():
    if not glfw.init():
        return
    window = glfw.create_window(480, 480, "2020098240-2-2", None, None)
    if not window:
        glfw.terminate()
        return
    
    # settings
    glfw.make_context_current(window=window)
    glfw.swap_interval(1)

    # main render loop
    while not glfw.window_should_close(window=window):
        glfw.poll_events()
        theta = glfw.get_time()
        R = np.array([[np.cos(theta), -np.sin(theta), 0.],
                      [np.sin(theta),  np.cos(theta), 0.],
                      [            0,              0, 1.]])
        T = np.identity(3)
        T[0, 2] = 0.5
        render(R@T)
        glfw.swap_buffers(window=window)
    glfw.terminate()
    return

if __name__ == "__main__":
    main()

