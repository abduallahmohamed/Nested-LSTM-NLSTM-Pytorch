import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_dim, bias=True):
        """
        Initialize LSTM cell.
        
        Parameters
        ----------
        input_size: int
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.



        """

        super(LSTMCell,self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.Gates = nn.Linear(self.input_size + self.hidden_dim, 4 * self.hidden_dim,bias =self.bias)

    def forward(self, input_tensor, cur_state):


        h_cur, c_cur = cur_state

        stacked_inputs = torch.cat([input_tensor, h_cur],dim= 1)
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
        cell = (remember_gate * c_cur) + (in_gate * cell_gate)
        hidden = out_gate * F.tanh(cell)

        return hidden, cell
    
    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim)).cuda())
    
class NestedLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_dim,bias = True):
        super(NestedLSTMCell,self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.Gates = nn.Linear(self.input_size + self.hidden_dim, 4 * self.hidden_dim,bias=self.bias)
        self.cell_core = LSTMCell(self.hidden_dim,self.hidden_dim,self.bias) 


    def forward(self, input_tensor, total_state):

        cur_state,cell_state = total_state
        h_cur,c_cur  = cur_state


        stacked_inputs = torch.cat((input_tensor, h_cur), 1)
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
        cell = (remember_gate * c_cur) + (in_gate * cell_gate)
#         print(cell.size())

        
        cell_state = self.cell_core(cell,cell_state)
        cell = cell_state[0]
        hidden = out_gate * F.tanh(cell) 

        return ((hidden,cell),cell_state)
        
    def init_hidden(self, batch_size):
        cell_state = (Variable(torch.zeros(batch_size, self.hidden_dim)).cuda(),
                        Variable(torch.zeros(batch_size, self.hidden_dim)).cuda())

        return (cell_state,self.cell_core.init_hidden(batch_size))
    
class NestedLSTM(nn.Module):

    def __init__(self, input_size,hidden_dim, num_layers = 1,
                 batch_first=False, bias=True, return_all_layers=False):
        
        
        super(NestedLSTM, self).__init__()


        # Make sure that  `hidden_dim` are lists having len == num_layers
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_size = input_size

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):

            cell_list.append(NestedLSTMCell(input_size=self.input_size,
                                          hidden_dim=self.hidden_dim[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c) or (b, t, c)
        hidden_state: todo
            None. todo implement stateful
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c) -> (b, t, c)
            input_tensor = input_tensor.permute(1, 0, 2)


        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            state = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):

                state = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :],
                                                 total_state=state)
                output_inner.append(state[0][0])

            layer_output = torch.stack(output_inner, dim=1)
            
            cur_layer_input = layer_output
            if not self.batch_first:
                layer_output_list.append(layer_output.permute(1, 0, 2))
            else:
                layer_output_list.append(layer_output)
            last_state_list.append(state)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list, hidden_state

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states


    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
