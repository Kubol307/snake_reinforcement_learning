# import libraries
import matplotlib.pyplot as plt
import numpy as np
import math


# Definition of cycloid:
# x = r(t - sint)
# y = r(1 - cost)


# take input from user to define radius and number of rolls of the circle
r = int(input("Define circle radius:"))
iter = int(input("Define number of rolls:"))

# set list for x points of cycloid 
x = [r*(t - math.sin(t)) for t in np.arange(-iter*np.pi, iter*np.pi, 0.04)]

# define list of zeroes for y points of cycloid
y = [0] * len(x)

# define x starting position of circle's center and plot
x_start = r*(-iter*np.pi - math.sin(-iter*np.pi))

# define x and y position of green dot
x_point = x_start
y_point = 0

# create red circle 
circle1 = plt.Circle((x_start, r), r, color='r', fill=False)

# enable interactive mode
plt.ion()

# create figure and subplot
fig = plt.figure()
ax = fig.add_subplot(111)

# set x and y limits for the plot
plt.ylim([0, -2*x_start])
plt.xlim([x_start, -x_start])

# draw cycloid, green dot and circle on one plot
line1, = ax.plot(x, y, 'b-')
line2, = ax.plot(x_point, y_point, 'g.')
ax.add_patch(circle1)

i = 0
  
for t in np.arange(-iter*np.pi, iter*np.pi, 0.04):

    # change x values
    add_x = r*(t - math.sin(t))
    x[i:] = [add_x]*(len(x)-i)
    line1.set_xdata(x)

    # change y values
    add_y = r*(1 - math.cos(t))
    y[i:] = [add_y]*(len(y)-i)
    line1.set_ydata(y)

    # change current position for green dot
    line2.set_ydata(add_y)
    line2.set_xdata(r*(t - math.sin(t)))
    
    # move circle
    circle1.center = x_start+0.04*r*i, r

    i += 1

    # draw all plots and flush GUI 
    fig.canvas.draw()
    fig.canvas.flush_events()