from functools import reduce
class Node():
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream =[]
        self.upstream = []
        self.output = 0
        self.delta = 0
    
    def set_out_put(self, output):
        self.output = output 

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    def calc_output(self):
        import numpy as np
        sigmoid = lambda x: 1/(1+np.exp(-x))
        output=reduce(lambda ret, cnn: ret+cnn.upstream_node.output* cnn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(lambda ret, cnn: ret+cnn.downsream_node.delta*cnn.weight, self.downstream, 0.0)
        self.delta = self.output*(1-self.output)*downstream_delta

    def calc_output_layer_delta(self, label):
        self.delta = self.output*(1-self.output)*(label - self.output)

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' %(self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret +'\n\t'+str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' +str(conn), self.upstream, '')
        return node_str +'\n\tdownstream:' +downstream_str +'\n\tupstream:' + upstream_str


