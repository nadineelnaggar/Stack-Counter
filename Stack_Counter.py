import torch
import torch.nn as nn
import torch.autograd
from torch.autograd import Function
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import Adam

device = torch.cuda if torch.cuda.is_available() else 'cpu'

# all data (push and pop) is input as a single tensor

class StackCounter(Function):

    @staticmethod
    def forward(ctx, input, state):
        # print('pushh = ',input[0])
        # print('popp = ',input[1])
        # stack_depth = state[0].clone()
        # false_pop_count = state[1].clone()
        # if input[0]<0 and input[1]<0:
        #     input*=-1
        ctx.save_for_backward(input, state)
        # ctx.preop_stack_depth = stack_depth
        # ctx.falsepop = false_pop_count
        output = state.clone()

        #binarise push and pop
        push = 0
        pop = 0
        threshold_push = 0.5
        threshold_pop = 0.5
        op = 'NoOp'
        if input[0]< threshold_push:
            push = 0
        elif input[0]>=threshold_push and input[0]<input[1]:
        # if input[0] >= threshold_push and input[0] < input[1]:
            push = 0
            pop = 1
            op = 'Pop'
        elif input[0]>=threshold_push and input[0]>=input[1]:
            push = 1
            pop = 0
            op='Push'
        if input[1]<threshold_pop:
            pop = 0
        elif input[1]>=threshold_pop and input[1]<input[0]:
        # if input[1] >= threshold_pop and input[1] < input[0]:
            pop = 0
            push = 1
            op = 'Push'
        elif input[1]>=threshold_pop and input[1]>input[0]:
            pop = 1
            push = 0
            op = 'Pop'
        if push==0 and pop==0:
            op='NoOp'
        # print('Push Input = ',input[0])
        # print('Push Threshold = ',threshold_push)
        # print('Pop Input = ',input[1])
        # print('Pop Threshold = ',threshold_pop)
        # print('Binary push = ',push)
        # print('Binary pop = ',pop)
        # print('operation = ',op)

        #update stack state
        # if push == 1 and pop == 0:
        #     stack_depth+=1
        #     print('PUSH OPERATION TRIGGERED')
        # elif push == 0 and pop ==1:
        #     if stack_depth>0:
        #         stack_depth+= -1
        #     elif stack_depth==0:
        #         false_pop_count+=1
        #     print('POP OPERATION TRIGGERED')
        # elif push==0 and pop == 0:
        #     print('NOOP OPERATION TRIGGERED')
        #     pass

        if op == 'Push':
            # stack_depth+=1
            output[0]=state[0]+1
            # print('PUSH OPERATION TRIGGERED')
        elif op=='Pop':
            if state[0]>0:
                output[0] = state[0]-1
            # if stack_depth>0:
            #     stack_depth+= -1
            elif state[0]==0:
                output[1]=state[1]+1
            # elif stack_depth==0:
            #     false_pop_count+=1
            # print('POP OPERATION TRIGGERED')
        elif op=='NoOp':
            # print('NOOP OPERATION TRIGGERED')
            pass

        ctx.op = op
        # print('Activation FW stack depth = ', stack_depth)
        # print('Activation FW false pop count = ', false_pop_count)
        # output = torch.cat((stack_depth,false_pop_count),dim=0)
        # print('output tensor cat test')
        # output = torch.tensor([stack_depth, false_pop_count], requires_grad=True)
        ignore = torch.tensor([0,0],dtype=torch.float32)

        return output
        # return state


    # @staticmethod
    # def forward(ctx, input, state):
    #     # input = input.squeeze(0)
    #     # input[2].requires_grad_(False)
    #     # input[3].requires_grad_(False)
    #     push = input[0].clone()
    #     pop = input[1].clone()
    #     print("pushh = ",push)
    #     print('popp = ',pop)
    #     # stack_depth = input[2].clone()
    #     # false_pop_count = input[3].clone()
    #     stack_depth = state[0].clone()
    #     false_pop_count = state[1].clone()
    #     ctx.save_for_backward(input, state)
    #     # ctx.save_for_backward(push, pop)
    #     # ctx.false_pop_count = false_pop_count
    #     ctx.preop_stack_depth = stack_depth
    #     ctx.falsepop = false_pop_count
    #     if push > pop and push>=0.5:
    #         stack_depth += 1
    #     elif pop > push and pop>=0.5:
    #         if stack_depth > 0:
    #             stack_depth += -1
    #         elif stack_depth == 0:
    #             false_pop_count += 1
    #     elif push == pop:
    #         pass
    #     print('Activation FW stack depth = ', stack_depth)
    #     print('Activation FW false pop count = ', false_pop_count)
    #     # output = Variable(torch.tensor([stack_depth,false_pop_count]),requires_grad=True)
    #     output = torch.tensor([stack_depth,false_pop_count],requires_grad=True)
    #     # print('ctx preop stack depth in forward pass = ',ctx.preop_stack_depth)
    #     return output#.to(device)
    #     # return torch.tensor([stack_depth, false_pop_count])


    @staticmethod
    def backward(ctx, grad_output):
        # grad_output = grad_output.shape[0]
        # grad_input = None

        # print('grad_output = ', grad_output)
        # print('grad_output size = ',grad_output.shape)

        grad_output_stack_depth = grad_output[0].clone().detach()
        grad_output_falsepop = grad_output[1].clone().detach()
        grad_push_stack_depth = torch.tensor(1, dtype=torch.float32)
        grad_push_falsepop = torch.tensor(0, dtype=torch.float32)
        # push, pop = ctx.saved_tensors
        input, state = ctx.saved_tensors

        # print('previous state[0] = preop stack depth = ',state[0])
        # print('previous state[1] = preop false pop count = ',state[1])

        grad_input = grad_output.clone()

        # print('grad_input before calculations = ',grad_input)
        # print('grad_input.shape', grad_input.shape)
        # # print('preop stack depth = ',ctx.preop_stack_depth)
        # print('new stack depth = ',ctx.preop_stack_depth)
        # print('new false pop count = ',ctx.falsepop)

        # if ctx.preop_stack_depth == 0:
        if state[0]==0:
            grad_pop_stack_depth = torch.tensor(0, dtype=torch.float32)
            grad_pop_falsepop = torch.tensor(1, dtype=torch.float32)
        else:
            grad_pop_stack_depth = torch.tensor(-1, dtype=torch.float32)
            grad_pop_falsepop = torch.tensor(0, dtype=torch.float32)

        # multiply input gradients by output gradients and return the correct one (4 cases) and return 2 values
        # grad_pop = Variable(grad_pop_stack_depth*grad_output[0] + grad_pop_falsepop*grad_output[1],requires_grad=True)
        # grad_push = Variable(grad_push_stack_depth*grad_output[0]+grad_push_falsepop*grad_output[1],requires_grad=True)


        # grad_pop = torch.tensor(grad_pop_stack_depth*grad_input[0] + grad_pop_falsepop*grad_input[1])
        # grad_push = torch.tensor(grad_push_stack_depth*grad_input[0] + grad_push_falsepop*grad_input[1])

        # print('grad_push_stack_depth = ', grad_push_stack_depth)
        # print('grad_push_falsepop = ',grad_push_falsepop)
        # print('grad_pop_stack_depth = ',grad_pop_stack_depth)
        # print('grad_pop_falsepop = ',grad_pop_falsepop)

        grad_pop = (grad_pop_stack_depth*grad_input[0]) + (grad_pop_falsepop*grad_input[1])
        grad_push = (grad_push_stack_depth*grad_input[0]) + (grad_push_falsepop*grad_input[1])

        # print('dL/dPush = dSD/dPush * dL/dSD + dFPC/dPush * dL/dFPC = [',grad_push_stack_depth,'*',grad_input[0],'] + [',grad_push_falsepop,'*',grad_input[1],'] = ',grad_push_stack_depth*grad_input[0],' + ',grad_push_falsepop*grad_input[1],' = ',grad_push)
        # print('dL/dPop = dSD/dPop * dL/dSD + dFPC/dPop * dL/dFPC = [',grad_pop_stack_depth,'*',grad_input[0],'] + [',grad_pop_falsepop,'*',grad_input[1],'] = ',grad_pop_stack_depth*grad_input[0],' + ',grad_pop_falsepop*grad_input[1],' = ',grad_pop)
        #
        # print('Stack Backward grad_push = ',grad_push)
        # print('Stack Backward grad_pop = ',grad_pop)


        # grad_pop = Variable(grad_pop_falsepop * grad_output_falsepop + grad_pop_stack_depth * grad_output_stack_depth,requires_grad=True)
        # grad_push = Variable(grad_push_stack_depth * grad_output_stack_depth + grad_push_falsepop * grad_output_falsepop,requires_grad=True)
        # grad_input = Variable(torch.tensor([grad_push, grad_pop]),requires_grad=True)



        grad_input = torch.tensor([grad_push, grad_pop], requires_grad=True)

        grad_stackdepth_out_stackdepth_in = 0
        grad_falsepop_out_falsepop_in = 0
        grad_stackdepth_out_falsepop_in = 0
        grad_falsepop_out_stackdepth_in = 0

        if ctx.op == 'Push':
            grad_stackdepth_out_stackdepth_in = 1
            grad_falsepop_out_falsepop_in = 1
            grad_stackdepth_out_falsepop_in = 0
            grad_falsepop_out_stackdepth_in = 0
        elif ctx.op == 'NoOp':
            grad_stackdepth_out_stackdepth_in = 1
            grad_falsepop_out_falsepop_in = 1
            grad_stackdepth_out_falsepop_in = 0
            grad_falsepop_out_stackdepth_in = 0
        elif ctx.op == 'Pop':
            if state[0]==0: #if stack is empty
                grad_stackdepth_out_stackdepth_in=0
                grad_falsepop_out_falsepop_in=1
                grad_falsepop_out_stackdepth_in=-1
                grad_stackdepth_out_falsepop_in=0
            elif state[0]>0: # if stack is not empty
                grad_stackdepth_out_stackdepth_in = 1
                grad_falsepop_out_falsepop_in = 1
                grad_stackdepth_out_falsepop_in = 0
                grad_falsepop_out_stackdepth_in = 0
        # grad_y = None
        grad_state = grad_output.clone()
        # grad_state_stackdepth=torch.tensor(0,dtype=torch.float32)
        # grad_state_falsepop=torch.tensor(0,dtype=torch.float32)
        grad_state_stackdepth = (grad_stackdepth_out_stackdepth_in*grad_state[0])+(grad_falsepop_out_stackdepth_in*grad_state[1])
        grad_state_falsepop = (grad_stackdepth_out_falsepop_in*grad_state[0])+(grad_falsepop_out_falsepop_in*grad_state[1])
        grad_state = torch.tensor([grad_state_stackdepth,grad_state_falsepop],requires_grad=True)


        # grad_input[0] = grad_push
        # grad_input[1] = grad_pop
        # grad_input[2] = torch.tensor(0,dtype=torch.float32)
        # grad_input[3] = torch.tensor(0,dtype=torch.float32)


        # grad_input = torch.tensor([grad_push, grad_pop])
        # grad_input = Variable(torch.tensor([grad_push,grad_pop]), requires_grad=True)
        # print('grad_input = ',grad_input)
        #
        # print('###########################################################')
        # print('grad_output = ',grad_output)
        # print('grad_input = ',grad_input)
        # print('grad_state = ',grad_state)
        # if testCorrectness(grad_input,grad_output) == True and testCorrectness(grad_state,grad_output)==True:
        #     print('grad_input = ',grad_input,'\ngrad_output from above = ',grad_output,'\ngrad_input is correct')

        """
        test if the resulting gradients are correct
        this is only used for unit testing.
        comment lines between ######### if not unit testing 
        from this point up until the return (do not comment the return)
        """

        #####################################################################################

        # grad_state_correct = False
        # grad_input_correct = False
        #
        # if state[0]==0:
        #     if grad_input[0] == grad_output[0] and grad_input[1]==grad_output[1]:
        #         print('grad_out == ',grad_output)
        #         print('grad_input == ',grad_input)
        #         print('executed operation = ',ctx.op)
        #         print('grad_input is correct in this ',ctx.op,'operation')
        #         grad_input_correct=True
        #     else:
        #         print('grad_out == ', grad_output)
        #         print('grad_input == ', grad_input)
        #         print('grad_input is incorrect in this ',ctx.op,'operation')
        #         grad_input_correct=False
        #     if (ctx.op=='Push' or ctx.op=='NoOp'):
        #         if grad_state[0]==grad_output[0] and grad_state[1] == grad_output[1]:
        #             print('grad_out == ',grad_output)
        #             print('grad_state == ',grad_state)
        #             print('executed operation = ',ctx.op)
        #             print('grad_state is correct in this ',ctx.op, 'operation')
        #             grad_state_correct=True
        #         else:
        #             print('grad_out == ', grad_output)
        #             print('grad_state == ', grad_state)
        #             print('executed operation = ', ctx.op)
        #             print('grad_state is incorrect in this ', ctx.op, 'operation')
        #             grad_state_correct=False
        #     elif ctx.op == 'Pop':
        #         if grad_state[0]==(grad_output[1]*-1) and grad_state[1] == grad_output[1]:
        #             print('grad_out == ',grad_output)
        #             print('grad_state == ',grad_state)
        #             print('executed operation = ',ctx.op)
        #             print('grad_state is correct in this ',ctx.op, 'operation')
        #             grad_state_correct=True
        #         else:
        #             print('grad_out == ', grad_output)
        #             print('grad_state == ', grad_state)
        #             print('executed operation = ', ctx.op)
        #             print('grad_state is incorrect in this ', ctx.op, 'operation')
        #             grad_state_correct=False
        # elif state[0]>0:
        #     if grad_input[0] == grad_output[0] and grad_input[1]==(grad_output[0]*-1):
        #         print('grad_out == ',grad_output)
        #         print('grad_input == ',grad_input)
        #         print('executed operation = ',ctx.op)
        #         print('grad_input is correct in this ',ctx.op,'operation')
        #         grad_input_correct=True
        #     else:
        #         print('grad_out == ', grad_output)
        #         print('grad_input == ', grad_input)
        #         print('grad_input is incorrect in this ',ctx.op,'operation')
        #         grad_input_correct=False
        #     if (ctx.op=='Push' or ctx.op=='NoOp'):
        #         if grad_state[0]==grad_output[0] and grad_state[1] == grad_output[1]:
        #             print('grad_out == ',grad_output)
        #             print('grad_state == ',grad_state)
        #             print('executed operation = ',ctx.op)
        #             print('grad_state is correct in this ',ctx.op, 'operation')
        #             grad_state_correct=True
        #         else:
        #             print('grad_out == ', grad_output)
        #             print('grad_state == ', grad_state)
        #             print('executed operation = ', ctx.op)
        #             print('grad_state is incorrect in this ', ctx.op, 'operation')
        #             grad_state_correct=False
        #     elif ctx.op == 'Pop':
        #         if grad_state[0]==grad_output[0] and grad_state[1] == grad_output[1]:
        #             print('grad_out == ',grad_output)
        #             print('grad_state == ',grad_state)
        #             print('executed operation = ',ctx.op)
        #             print('grad_state is correct in this ',ctx.op, 'operation')
        #             grad_state_correct=True
        #         else:
        #             print('grad_out == ', grad_output)
        #             print('grad_state == ', grad_state)
        #             print('executed operation = ', ctx.op)
        #             print('grad_state is incorrect in this ', ctx.op, 'operation')
        #             grad_state_correct=False
        #
        # if grad_state_correct==True and grad_input_correct==True:
        #     print('BOTH INPUT AND STATE GRADIENTS ARE CORRECT')
        # elif grad_input_correct==True and grad_state_correct==False:
        #     print('STATE GRADIENT INCORRECT')
        # elif grad_input_correct==False and grad_state_correct==True:
        #     print('INPUT GRADIENTS INCORRECT')
        # elif grad_input_correct==False and grad_state_correct==False:
        #     print('INCORRECT INPUT AND STATE GRADIENTS')

    ########################################################################################

        return grad_input, grad_state  # .to(device)

        """
        push = 1 --> grad_push = 1
        push = 0 --> grad_push = 0
        pop = 1, stack_depth>0, false_pop_count=0 --> grad_pop_stack_depth = -1 grad_pop_falsepop = 0
        pop = 1, stack_depth = 0, false_pop_count > 0 --> grad_stack_depth = 0, grad_pop_falsepop = 1


        """

    # def testCorrectness(ctx,grad_in, grad_out):
    #     input, state = ctx.saved_tensors
    #     if ctx.op == 'Push':
    #         pass



class StackCounterNN(nn.Module):
    def __init__(self):
    # def __init__(self, recurrent=False):
        super(StackCounterNN, self).__init__()
    #     super().__init__()
        # self.stack_depth = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        # self.false_pop_count = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        # self.state = torch.tensor([0,0],dtype=torch.float32,requires_grad=True)
        # # self.stack_depth = torch.tensor(self.state[0],dtype=torch.float32,requires_grad=False)
        # # self.false_pop_count = torch.tensor(self.state[1],dtype=torch.float32,requires_grad=False)
        self.stack_depth = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        self.false_pop_count = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        # # self.state = torch.tensor([self.stack_depth,self.false_pop_count],dtype=torch.float32,requires_grad=True)
        # self.state = torch.tensor([0,0],dtype=torch.float32,requires_grad=True)
        self.stack = StackCounter.apply
        # self.recurrent=recurrent

    def reset(self):
        self.stack_depth = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        self.false_pop_count = torch.tensor(0, dtype=torch.float32, requires_grad=False)
        # self.state[0]=0
        # self.state[1]=0

    def forward(self, x,y=None):
    # def forward(self, x, y=None):
        # x = x.squeeze(0)
        # push = x[0]
        # # push.requires_grad_(True)
        # pop = x[1]
        # pop.requires_grad_(True)

        # print('push = ', push, 'pop = ', pop)

        # y = torch.cat([push,pop,self.stack_depth,self.false_pop_count])
        # x = Variable(torch.tensor([x[0],x[1],self.stack_depth,self.false_pop_count]), requires_grad=True)#adding requires_grad = True causes warnings®
        # x = Variable(torch.FloatTensor([x[0], x[1], self.stack_depth, self.false_pop_count]), requires_grad=True)

        # x = torch.tensor([[x[0],x[1],self.stack_depth,self.false_pop_count]], requires_grad=True)

        # x = torch.tensor([x[0], x[1], self.stack_depth, self.false_pop_count], requires_grad=True)
        if y==None:
            y = torch.tensor([self.stack_depth, self.false_pop_count], requires_grad=True)

        # y = torch.tensor([self.stack_depth, self.false_pop_count],requires_grad=True)
        # x = torch.tensor([push, pop, self.stack_depth, self.false_pop_count])

        # x = torch.tensor([x[0], x[1], self.stack_depth, self.false_pop_count])
        # push.requires_grad = True
        # pop.requires_grad = True
        # x = torch.tensor([push,pop,self.stack_depth,self.false_pop_count], requires_grad=True)

        # print(x)

        # print(y.shape)
        # x = self.stack(torch.tensor([x[0],x[1],self.stack_depth,self.false_pop_count]))
        x = self.stack(x,y)
        # print('stack params = ',list(self.parameters()))
        # self.register_backward_hook(x)
        # x = self.stack(x,self.state)
        # self.state = x

        self.reset()
        self.stack_depth+=x[0]
        self.false_pop_count+=x[1]

        # self.stack_depth = x[0]
        # self.false_pop_count = x[1]
        # print('NN FW pass output stack depth = ',self.stack_depth,'NN FW pass output false pop count = ', self.false_pop_count)
        # print('grad_function of x = ',x.grad_fn)
        # print('grad_fn of x[0] (push) = ',push.grad_fn)
        # print('grad_fn of x[1] (pop) = ',pop.grad_fn)
        # print('stack depth grad function = ',self.stack_depth.grad_fn)
        # print('false pop grad function = ',self.false_pop_count.grad_fn)
        # print('push grad function = ',push.grad_fn)
        return x
    # def reset(self):
    #     self.stack_depth = torch.tensor(0, dtype=torch.float32, requires_grad=False)
    #     self.false_pop_count = torch.tensor(0, dtype=torch.float32, requires_grad=False)
    #     # self.state[0]=0
    #     # self.state[1]=0
    def editStackState(self, new_stack_depth, new_false_pop_count):
        if new_stack_depth<0:
            self.stack_depth = torch.tensor(0,dtype=torch.float32,requires_grad=False)
        else:
            self.stack_depth = torch.tensor(new_stack_depth, dtype=torch.float32, requires_grad=False)
        if new_false_pop_count<0:
            self.false_pop_count = torch.tensor(0,dtype=torch.float32,requires_grad=False)
        else:
            self.false_pop_count = torch.tensor(new_false_pop_count, dtype=torch.float32, requires_grad=False)
        # self.stack_depth = torch.tensor(new_stack_depth,dtype=torch.float32, requires_grad=False)
        # self.false_pop_count = torch.tensor(new_false_pop_count,dtype=torch.float32,requires_grad=False)


# stack = StackCounterNN()
# #original
# push = torch.tensor([1,0],dtype=torch.float32)
# pop = torch.tensor([0,1],dtype=torch.float32)
# noOp1 = torch.tensor([1,1],dtype=torch.float32)
# noOp2 = torch.tensor([0,0],dtype=torch.float32)
#
# #edited
# # push = torch.tensor([[1,0]],dtype=torch.float32)
# # pop = torch.tensor([[0,1]],dtype=torch.float32)
# # noOp1 = torch.tensor([[1,1]],dtype=torch.float32)
# # noOp2 = torch.tensor([[0,0]],dtype=torch.float32)
#
#
#
# # push_empty = torch.tensor([1,0,0,0],dtype=torch.float32)
# # # print(StackCounterNN(push_empty))
# print(push.shape)
# print(push[0])
#
# output1 = stack(push)
# print('output1 = ',output1)
# # print(output1.grad)
# # print(output1 = stack(push))
# print(stack(push))
# print(stack(pop))
# print(stack(pop))
# print(stack(pop))
# print(stack(push))
# output2 = stack(pop)
# print(output2)
# # print(output2.grad)
# print('stack status after pop = ',stack(pop))
# print('stack status after push = ',stack(push))
#
# labels = Variable(torch.tensor([1,1],dtype=torch.float32),requires_grad=True)
# output2 = Variable(output2,requires_grad=True)
# # labels = torch.tensor([1,1],dtype=torch.float32)
# gradients_push = torch.tensor([1,0],dtype=torch.float32)
# gradients_pop = torch.tensor([-1,0],dtype=torch.float32)
# gradients_falsepop = torch.tensor([0,1],dtype=torch.float32)
#
# criterion = nn.MSELoss()
# loss = criterion(output2,labels)
# print('loss = ',loss)
# # with torch.no_grad():
# # optimiser = optim.Adam(stack.parameters(),lr=0.001)
# # optimiser.zero_grad()
# # stack.zero_grad()
# print('loss backward = ',loss.backward())
# # print(stack.backward())
# print('output2 grad function = ',output2.grad_fn)
# print('output2 grad = ',output2.grad)
#
# output1 = Variable(output1, requires_grad=True)
# print('output1 grad before loss = ',output1.grad)
# labels1 = Variable(torch.tensor([1,0],dtype=torch.float32),requires_grad=True)
# loss = criterion(output1, labels1)
# print('loss = ',loss)
# print('loss backward = ',loss.backward())
# print('grad after push = ',output1.grad)
#
# print('stack parameters = ',stack.parameters())
# print('output1 grad = ',output1.grad)
# print('output1 grad fn = ',output1.grad_fn)

# model = StackCounterNN()
# lr = 0.001
# optimiser = optim.Adam(model.parameters(), lr=lr)


#try a test module that incorporates the stack and see if it will work with backward function


# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.stack = StackCounterNN()

