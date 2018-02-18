import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable


#Following a similar CONVLSTM implementation from https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py
class LSTMCell(nn.Module):
    """
    Generate a minimal LSTM cell
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        #spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] #+ list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = F.sigmoid(in_gate)
        remember_gate = F.sigmoid(remember_gate)
        out_gate = F.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = F.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * F.tanh(cell)

        return hidden, cell


#NLSTM
class NLSTM(nn.Module):


    def __init__(self, input_size, hidden_size):
        super(NLSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
        self.cell_core = LSTMCell(hidden_size,hidden_size)
        self.cell_state = None

    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        #spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] #+ list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = F.sigmoid(in_gate)
        remember_gate = F.sigmoid(remember_gate)
        out_gate = F.sigmoid(out_gate)


        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
#         print(cell.size())
        self.cell_state = self.cell_core(cell,self.cell_state)
        cell = self.cell_state[0]
        hidden = out_gate * F.tanh(cell)
        return hidden, cell
        
        
#Test
# define batch_size, channels, height, width
b, c = 1, 3
d = 5           # hidden state size
lr = 1e-1       # learning rate
T = 6           # sequence length
max_epoch = 50000  # number of epochs

# set manual seed
torch.manual_seed(0)

print('Instantiate model')
model = NLSTM(c,d)
print(repr(model))

print('Create input and target Variables')
x = Variable(torch.rand(T, b, c))
y = Variable(torch.randn(T, b, d))

print('Create a MSE criterion')
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)
lambda2 = lambda epoch: 0.99 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=lambda2)
print('Run for', max_epoch, 'iterations')
for epoch in range(0, max_epoch):
    state = None
    loss = 0
    for t in range(0, T):
        state = model(x[t], state)
        loss += loss_fn(state[0], y[t])

    print(' > Epoch {:2d} loss: {:.3f}'.format((epoch+1), loss.data[0]))

    # zero grad parameters
    optimizer.zero_grad()
    # compute new grad parameters through time!
    loss.backward(retain_graph=True)

    # learning_rate step against the gradient
    optimizer.step()
#     scheduler.step()
    print('\nLR ={}'.format(optimizer.param_groups[0]['lr']))


print('Input size:', list(x.data.size()))
print('Target size:', list(y.data.size()))
print('Last hidden state size:', list(state[0].size()))
