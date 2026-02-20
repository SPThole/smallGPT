import math
from graphviz import Digraph

#think about best class structure and methods for backwards and grad


class Tensor:
    def __init__(self,value):
        self.value = value
        self.grad = 0.0
        self._backwards = lambda: None
        self.inputs = []
        self.operation = None
        # self.name = name
    
    def __add__(self,tensor_to_add):
        if isinstance(tensor_to_add,Tensor):
            output = Tensor(self.value+tensor_to_add.value)
        else:
            tensor_to_add = Tensor(tensor_to_add)
            output = Tensor(self.value+tensor_to_add.value)
        def _backwards():
            self.grad += 1.0 * output.grad 
            if isinstance(tensor_to_add,Tensor):
                # current grad contrib * earlier grad
                tensor_to_add.grad += 1.0 * output.grad
            else:
                tensor_to_add.grad += 0.0
        output._backwards = _backwards
        output.inputs = [self,tensor_to_add]
        output.operation = '+'
        return output
    
    def __mul__(self,tensor_to_mult):
        if isinstance(tensor_to_mult,Tensor):
            output = Tensor(self.value*tensor_to_mult.value)
        else:
            tensor_to_mult = Tensor(tensor_to_mult)
            output = Tensor(self.value*tensor_to_mult.value)
        def _backwards():
            if isinstance(tensor_to_mult,Tensor):
                self.grad += tensor_to_mult.value * output.grad
                tensor_to_mult.grad += self.value * output.grad
               
            else:
                self.grad += tensor_to_mult * output.grad# current grad contrib * earlier grad
                tensor_to_mult.grad = 0.0
        output.inputs = [self,tensor_to_mult]
        output._backwards = _backwards
        output.operation = '*'
        return output
    
    def __rmul__(self,tensor_to_mult):
        if isinstance(tensor_to_mult,Tensor):
            output = Tensor(self.value*tensor_to_mult.value)
        else:
            output = Tensor(self.value*tensor_to_mult)
            tensor_to_mult = Tensor(tensor_to_mult)
        def _backwards():
            if isinstance(tensor_to_mult,Tensor):
                self.grad += tensor_to_mult.value * output.grad
                tensor_to_mult.grad += self.value * output.grad
              
            else:
                self.grad += tensor_to_mult * output.grad# current grad contrib * earlier grad
                tensor_to_mult.grad = 0.0
        output.inputs = [self,tensor_to_mult]
        output._backwards = _backwards
        output.operation = '*'
        return output
    
    def __pow__(self,power):
        if isinstance(power,int) or isinstance(power,float):
            output = Tensor(self.value**power)
        else:
            raise Exception
        
        def _backwards():
            self.grad += power*self.value**(power-1)*output.grad
        
        output.inputs = [self]
        output._backwards = _backwards
        output.operation = 'pow'
        return output
    
    def __sub__(self,tensor_to_sub):
        return self + -1*tensor_to_sub
    

    def backwards(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.inputs:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backwards()

    @staticmethod
    def trace(root):
        nodes = set()
        edges = set()

        def build(node):
            if node not in nodes:
                nodes.add(node)
                for parent in node.inputs:
                    edges.add((parent, node))
                    build(parent)

        build(root)
        return nodes, edges
    
    @staticmethod
    def draw_graph(root):
        dot = Digraph(format="png", graph_attr={'rankdir': 'LR'})
        nodes, edges = Tensor.trace(root)
        for n in nodes:
            uid = str(id(n))
            label = f"{n.value:.4f}"
            dot.node(uid, label=label, shape="circle")
            if n.operation:
                op_id = uid + n.operation
                dot.node(op_id, label=n.operation, shape="box")
                dot.edge(op_id, uid)
        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2.operation)
        return dot
    
    
            
if __name__ == "__main__":
    a = Tensor(3.0)
    b = Tensor(4.0)
    c = Tensor(5.0)
    d = a**2*c - a+b + 4 
    o = Tensor.trace(d)
    dot = Tensor.draw_graph(d)
    dot.render("computation_graph", view=True)
    d.backwards()
    print(a.grad,b.grad,c.grad)


    


    

    
    