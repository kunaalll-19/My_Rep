import torch


#ASSIGNING RANDOM WEIGHTS
weight_1 = torch.rand((3,2),requires_grad=True)
weight_2 = torch.rand((3,3),requires_grad=True)
weight_3 = torch.rand((1,3),requires_grad=True)

#ASSIGNING RANDOM BIASES
bias_1 = torch.rand((1,3),requires_grad=True)
bias_2 = torch.rand((1,3),requires_grad=True)
bias_3 = torch.rand((1,1),requires_grad=True)


#ASSIGNING INPUT
inp = torch.rand((1,2),requires_grad=False)

#FORWARD PASS(INPUT X WRIGHT + BIAS)
def forward_pass_relu(inp,weight,bias):
    out=inp@weight.T+bias
    return torch.relu(out)

def forward_pass_sigmoid(inp,weight,bias):
    out=inp@weight.T+bias
    return torch.sigmoid(out)

pass_1=forward_pass_relu(inp,weight_1,bias_1)
pass_2=forward_pass_relu(pass_1,weight_2,bias_2)
pass_3=forward_pass_sigmoid(pass_2,weight_3,bias_3)

#TRUE OUTPUTS
out_true_1=torch.rand((1,3),requires_grad=False)
out_true_2=torch.rand((1,3),requires_grad=False)
out_true_3=torch.rand((1,1),requires_grad=False)


#CALCULATING LOSS
loss_1=torch.mean((pass_1-out_true_1)**2)
loss_2=torch.mean((pass_2-out_true_2)**2)
loss_3=torch.mean((pass_3-out_true_3)**2)

#TAKING POLYNOMIAL X**2+3*X+5
x=torch.rand(1,requires_grad=True)
print(x)

y=x**2+3*x+5
y.backward()
print(x.grad)

#BACKWARD PROPAGATION
loss_3.backward(retain_graph=True)
with torch.no_grad():
    weight_3=weight_3-0.01*weight_3.grad
    bias_3=bias_3-0.01*bias_3.grad
    weight_2=weight_2-0.01*weight_2.grad
    bias_2=bias_2-0.01*bias_2.grad
    weight_1=weight_1-0.01*weight_1.grad
    bias_1=bias_1-0.01*bias_1.grad
    