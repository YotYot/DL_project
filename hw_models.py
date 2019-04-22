import numpy as np
import random
from utils import int2bit, bit2int


#Below is a list of all operations chosen by the model.
#All the operations get a list of bits as inputs and outputs a list of bits.

class and_gate():
    inputs = 2
    outputs = 1

    @staticmethod
    def result(a):
        return a[0] & a[1]

class or_gate():
    inputs = 2
    outputs = 1

    @staticmethod
    def result(a):
        return a[0] | a[1]

class not_gate():
    inputs = 1
    outputs = 1

    @staticmethod
    def result(a):
        if a[0] == 0:
            return 1
        else:
            return 0

class mux():
    inputs = 3
    outputs = 1

    @staticmethod
    def result(a):
        return a[1] if a[2] else a[0]

class two_bits_mux():
    inputs = 5
    outputs = 2

    @staticmethod
    def result(a):
        return a[:2] if a[4] else a[2:4]

class two_bits_mult():
    inputs = 4
    outputs = 4

    @staticmethod
    def result(a):
        b = bit2int(a[:2])
        c = bit2int(a[2:])
        m = b*c
        return int2bit(m,4)

class three_bits_mult():
    #Max input is 7, max output 49 (6b)
    inputs = 6
    outputs = 6

    @staticmethod
    def result(a):
        b = bit2int(a[:3])
        c = bit2int(a[3:])
        m = b * c
        return int2bit(m,6)

class four_bits_mult():
    #Max input is 15, max output 225 (8b)
    inputs = 8
    outputs = 8

    @staticmethod
    def result(a):
        b = bit2int(a[:4])
        c = bit2int(a[4:])
        m = b * c
        return int2bit(m,8)

class adder():
    inputs = 2
    outputs = 2

    @staticmethod
    def result(a):
        b = bit2int(a[:1])
        c = bit2int(a[1:])
        m = b + c
        return int2bit(m, 2)

class two_bit_adder():
    #Inputs are between 0-3, max output is 6 (3bit)
    inputs = 4
    outputs = 3

    @staticmethod
    def result(a):
        b = bit2int(a[:2])
        c = bit2int(a[2:])
        m = b + c
        return int2bit(m, 3)

class three_bit_adder():
    #Inputs are between 0-7, max output is 14 (4bit)
    inputs = 6
    outputs = 4

    @staticmethod
    def result(a):
        b = bit2int(a[:3])
        c = bit2int(a[3:])
        m = b + c
        return int2bit(m, 4)


class model_generater():
    def __init__(self, inputs=3, depth=5):
        # classes_list = [and_gate, or_gate, not_gate, mux, two_bits_mux, two_bits_mult, three_bits_mult, four_bits_mult, adder, two_bit_adder, three_bit_adder]
        classes_list = [and_gate, or_gate, not_gate, mux, two_bits_mux, two_bits_mult, adder, two_bit_adder]
        # classes_list = [two_bits_mult,not_gate]
        # layer_inputs = list()
        self.layer_outputs = list()
        self.model = list()
        self.inputs = inputs
        for i in range(depth):
           current_layer_outputs = 0
           layer_ops = list()
           if i == 0:
               layer_inputs = inputs
           else:
               layer_inputs = self.layer_outputs[-1]
           while (layer_inputs > 0):
                optional_classes = [b for b in classes_list if b.inputs <= layer_inputs]
                chosen_op = random.choice(optional_classes)
                layer_inputs -= chosen_op.inputs
                current_layer_outputs += chosen_op.outputs
                layer_ops.append(chosen_op)
           self.layer_outputs.append(current_layer_outputs)
           self.model.append(layer_ops)
        self.outputs = current_layer_outputs
        print (self.model)
        print (self.outputs)

    def forward(self, inputs):
        assert (len(inputs) == self.inputs)
        inputs_for_next_layer = inputs
        outputs = list()
        for layer in self.model:
            i = 0
            for op in layer:
                res = op.result(inputs_for_next_layer[i:i+op.inputs])
                if isinstance(res, int):
                    outputs.append(res)
                else:
                    outputs.extend(res)
                i = i+op.inputs
            inputs_for_next_layer = outputs
            outputs = list()
        return inputs_for_next_layer

if __name__ == '__main__':
    model = model_generater(inputs=4, depth=2)
    print (model.forward([1,1,0,1]))




