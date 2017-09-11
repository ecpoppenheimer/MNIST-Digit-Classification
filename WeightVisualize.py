import EricNetPublic
import numpy as np
import matplotlib.pyplot as plt
import mnist_loader
import sys

# load the net
try:
    net = EricNetPublic.Network(sys.argv[1])
except IndexError:
    sys.exit("Please provide a file path that points to the location of the net you want to load as a command line  paramater to this script.")

# if you want to verify that the net works, load and process the training data
trainingData, validationData, testData = mnist_loader.load_data_wrapper()
print "Evaluation: {}/{}".format(net.evaluate(testData), len(testData))

i = 0

def onKey(event):
    global i
    if event.key == 'a':
        if i > 0:
            i -= 1
        else:
            return
    elif event.key == 'd':
        if i < len(net.weights) - 1:
            i += 1
        else:
            return
    else:
        return
        
    plt.imshow(net.weights[i], cmap='gray', interpolation='None')
    plt.title("layer {}.  Press 'a' and 'd' to cycle".format(i))
    plt.draw()

fig = plt.figure()    
plt.imshow(net.weights[0], cmap='gray', interpolation='None')
plt.title("Layer 0.  Press 'a' and 'd' to cycle")
fig.canvas.mpl_connect('key_press_event', onKey)
plt.show()
