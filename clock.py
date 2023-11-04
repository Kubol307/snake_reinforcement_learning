import numpy as np
import matplotlib.pyplot as plt
import time


# R = float(input('R:'))
R = 10
theta = 0
resolution = 30
x = [0]*30
y = [0]*30

x_minute = [0]*50
y_minute = [0]*50

plt.ion()
circle1 = plt.Circle((0, 0), R, color='black', fill=False)

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-')
line2, = ax.plot(x_minute, y_minute, 'blue-')
ax.add_patch(circle1)

plt.xlim([-1.5*R, 1.5*R])
plt.ylim([-1.5*R, 1.5*R])
for t in range(180):
    start_time = time.time()
    theta = 90 -t*6
    theta = 6.28*theta/360
    b_range = R*np.cos(theta)
    step = np.absolute(b_range)/resolution
    # print('b range:', b_range)
    index = 0

    for i in np.arange(min(0, b_range), max(0, b_range+step), step):
        # if len(x) <= resolution + 1:
        # x.append(i)
        # y.append(i*np.tan(theta))
        # else:
        x[index] = i
        y[index] = i*np.tan(theta)
        index += 1
        if index >= 30:
            break
    
    line1.set_xdata(x)
    line1.set_ydata(y)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    stop_time = time.time()
    time_passed = stop_time - start_time
    time.sleep(1 - time_passed)