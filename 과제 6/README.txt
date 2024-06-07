modified only SimpleScene.py

variables
- curvePoint:   clicked point
- isRunning:    True if animation running, else False
- startPoint:   curvePoint[0]
- clicked:      is cow clicked. True if 0 < #click
- isDrag:       cow state (drag)
- spotLen:      curvePoint length (6)
- maxLoop:      loop number (3)
- index:        index for curvePoint[]

functions
- display: modified
- onMouseButton: modified
- onMouseDrag: modified
- curve: added
- turnHead: added
- runAnimation: added
- start: added
- stop: added

logic:
- start() called when click 6 points
- start() set isRunning true
- then runAnimation() called by display()
- stop() called when animation end
- stop() set isRunning false
