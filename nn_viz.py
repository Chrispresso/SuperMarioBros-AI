from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QColor, QBrush
import sys
from typing import List
from neural_network import *
from mario import Mario
from config import Config

class NeuralNetworkViz(QtWidgets.QWidget):
    def __init__(self, parent, mario: Mario, size, config: Config):
        super().__init__(parent)
        self.mario = mario
        self.size = size
        self.config = config
        self.horizontal_distance_between_layers = 50
        self.vertical_distance_between_nodes = 10
        l = self.config.NeuralNetwork.hidden_layer_architecture + [6]
        self.num_neurons_in_largest_layer = max(l[1:])
        self.neuron_locations = {}
        self.tile_size = self.config.Graphics.tile_size
        self.neuron_radius = self.config.Graphics.neuron_radius

        # Set all neuron locations for layer 0 (Input) to be at the same point.
        # The reason I do this is because the number of inputs can easily become too many to show on the screen.
        # For this reason it is easier to not explicitly show the input nodes and rather show the bounding box of the rectangle.
        self.x_offset = 150 + 16//2*self.tile_size[0] + 5
        self.y_offset = 5 + 15*self.tile_size[1] + 5
        for nid in range(l[0]):
            t = (0, nid)
            
            self.neuron_locations[t] = (self.x_offset, self.y_offset)

        self.show()

    def show_network(self, painter: QtGui.QPainter):
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        painter.setRenderHints(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QtGui.QPainter.TextAntialiasing)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        painter.setPen(QPen(Qt.black, 1.0, Qt.SolidLine))
        horizontal_space = 20  # Space between Nodes within the same layer
        
        layer_nodes = self.mario.network.layer_nodes

        default_offset = self.x_offset
        h_offset = self.x_offset
        v_offset = self.y_offset + 50
        inputs = self.mario.inputs_as_array

        out = self.mario.network.feed_forward(inputs)

        active_outputs = np.where(out > 0.5)[0]
        max_n = self.size[0] // (2* self.neuron_radius + horizontal_space)
        
        # Draw nodes
        for layer, num_nodes in enumerate(layer_nodes[1:], 1):
            h_offset = (((max_n - num_nodes)) * (2*self.neuron_radius + horizontal_space))/2
            activations = None
            if layer > 0:
                activations = self.mario.network.params['A' + str(layer)]

            for node in range(num_nodes):
                x_loc = node * (self.neuron_radius*2 + horizontal_space) + h_offset
                y_loc = v_offset
                t = (layer, node)
                if t not in self.neuron_locations:
                    self.neuron_locations[t] = (x_loc + self.neuron_radius, y_loc)
                
                painter.setBrush(QtGui.QBrush(Qt.white, Qt.NoBrush))
                # Input layer
                if layer == 0:
                    # Is there a value being fed in
                    if inputs[node, 0] > 0:
                        painter.setBrush(QtGui.QBrush(Qt.green))
                    else:
                        painter.setBrush(QtGui.QBrush(Qt.white))
                # Hidden layers
                elif layer > 0 and layer < len(layer_nodes) - 1:
                    saturation = max(min(activations[node, 0], 1.0), 0.0)
                    painter.setBrush(QtGui.QBrush(QtGui.QColor.fromHslF(125/239, saturation, 120/240)))
                # Output layer
                elif layer == len(layer_nodes) - 1:
                    text = ('U', 'D', 'L', 'R', 'A', 'B')[node]
                    painter.drawText(h_offset + node * (self.neuron_radius*2 + horizontal_space), v_offset + 2*self.neuron_radius + 2*self.neuron_radius, text)
                    if node in active_outputs:
                        painter.setBrush(QtGui.QBrush(Qt.green))
                    else:
                        painter.setBrush(QtGui.QBrush(Qt.white))

                painter.drawEllipse(x_loc, y_loc, self.neuron_radius*2, self.neuron_radius*2)
            v_offset += 150

        # Reset horizontal offset for the weights
        h_offset = default_offset

        # Draw weights
        # For each layer starting at 1
        for l in range(2, len(layer_nodes)):
            weights = self.mario.network.params['W' + str(l)]
            prev_nodes = weights.shape[1]
            curr_nodes = weights.shape[0]
            # For each node from the previous layer
            for prev_node in range(prev_nodes):
                # For all current nodes, check to see what the weights are
                for curr_node in range(curr_nodes):
                    # If there is a positive weight, make the line blue
                    if weights[curr_node, prev_node] > 0:
                        painter.setPen(QtGui.QPen(Qt.blue))
                    # If there is a negative (impeding) weight, make the line red
                    else:
                        painter.setPen(QtGui.QPen(Qt.red))
                    # Grab locations of the nodes
                    start = self.neuron_locations[(l-1, prev_node)]
                    end = self.neuron_locations[(l, curr_node)]
                    # Offset start[0] by diameter of circle so that the line starts on the right of the circle
                    painter.drawLine(start[0], start[1] + self.neuron_radius*2, end[0], end[1])
        
        # Draw line straight down
        color = QColor(255, 0, 217)
        painter.setPen(QPen(color, 3.0, Qt.SolidLine))
        painter.setBrush(QBrush(Qt.NoBrush))

        x_start = 5 + 150 + (16/2 * self.tile_size[0])
        y_start = 5 + (15 * self.tile_size[1])
        x_end = x_start
        y_end = y_start + 5 + (2 * self.neuron_radius)
        painter.drawLine(x_start, y_start, x_end, y_end)

        # Set pen to be smaller and draw pink connections
        painter.setPen(QPen(color, 1.0, Qt.SolidLine))
        for nid in range(layer_nodes[1]):
            start = self.neuron_locations[(1, nid)]
            painter.drawLine(start[0], start[1], x_end, y_end)
        