import math

# every Unit corresponds to a wire in the diagrams
class Unit():
	def __init__(this, value, grad):
		# value computed in the forward pass
		this.value = value

		# the derivative of circuit output w.r.t (with respect to) this unit, computed in backward pass
		this.grad  = grad


class multiplyGate:

  def forward(this, u0 ,u1):

    # store pointers to input Units u0 and u1 and output unit utop
    this.u0 = u0
    this.u1 = u1
    this.utop = Unit(u0.value * u1.value, 0.0)
    return this.utop

  def backward(this):

    # take the gradient in output unit and chain it with the
    # local gradients, which we derived for multiply gate before
    # then write those gradients to those Units.

    this.u0.grad += this.u1.value * this.utop.grad
    this.u1.grad += this.u0.value * this.utop.grad
    	
    	
class addGate:
  def forward (this,u0, u1):
    this.u0 = u0 
    this.u1 = u1 # store pointers to input units
    this.utop = Unit(u0.value + u1.value, 0.0)
    return this.utop

  def backward(this):
    # add gate. derivative wrt both inputs is 1
    this.u0.grad += 1 * this.utop.grad;
    this.u1.grad += 1 * this.utop.grad;

class sigmoidGate:

  def sig(this,x):
    # helper function
    return 1 / (1 + math.exp(-x))

  def forward (this, u0):
    this.u0 = u0
    this.utop = Unit(this.sig(this.u0.value), 0.0)
    return this.utop;
  
  def backward(this):
    s = this.sig(this.u0.value)
    this.u0.grad += (s * (1 - s)) * this.utop.grad

# create input units
a = Unit(1.0, 0.0)
b = Unit(2.0, 0.0)
c = Unit(-3.0, 0.0)
x = Unit(-1.0, 0.0)
y = Unit(3.0, 0.0)

# create the gates
mulg0 = multiplyGate()
mulg1 = multiplyGate()
addg0 = addGate()
addg1 = addGate()
sg0 = sigmoidGate()

# do the forward pass
def forwardNeuron():
  ax = mulg0.forward(a, x) # a*x = -1
  by = mulg1.forward(b, y) # b*y = 6
  axpby = addg0.forward(ax, by) # a*x + b*y = 5
  axpbypc = addg1.forward(axpby, c) # a*x + b*y + c = 2
  s = sg0.forward(axpbypc) # 0.8808
  return s

fn = forwardNeuron()
print ("circuit output: " + str(fn.value))

fn.grad = 1.0
sg0.backward() # writes gradient into axpbypc
addg1.backward() # writes gradients into axpby and c
addg0.backward() # writes gradients into ax and by
mulg1.backward() # writes gradients into b and y
mulg0.backward() # writes gradients into a and x

step_size = 0.01
a.value += step_size * a.grad # a.grad is -0.105
b.value += step_size * b.grad # b.grad is 0.315
c.value += step_size * c.grad # c.grad is 0.105
x.value += step_size * x.grad # x.grad is 0.105
y.value += step_size * y.grad # y.grad is 0.210

print ('a.grad: {:.3f}\nb.grad: {:.3f}\nc.grad: {:.3f}\nx.grad: {:.3f}\ny.grad: {:.3f}'.format(a.grad, b.grad, c.grad, x.grad, y.grad))

fn = forwardNeuron()
print ("circuit output after one backprop: " + str(fn.value))

# checking the numerical gradient
a = 1
b = 2
c = -3
x = -1
y = 3
h = 0.0001
def forwardCircuitFast(a,b,c,x,y):
  return 1/(1 + math.exp(-(a*x + b*y + c)))

print ('a.grad: {:.3f}'.format((forwardCircuitFast(a+h,b,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h),
'\nb.grad: {:.3f}'.format((forwardCircuitFast(a,b+h,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h),
'\nc.grad: {:.3f}'.format((forwardCircuitFast(a,b,c+h,x,y) - forwardCircuitFast(a,b,c,x,y))/h),
'\nx.grad: {:.3f}'.format((forwardCircuitFast(a,b,c,x+h,y) - forwardCircuitFast(a,b,c,x,y))/h),
'\ny.grad: {:.3f}'.format((forwardCircuitFast(a,b,c,x,y+h) - forwardCircuitFast(a,b,c,x,y))/h))
