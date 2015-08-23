import os
import time
import numpy as np


from chainer import cuda, Variable, function, FunctionSet, optimizers
from chainer import functions as F

class VRAE(FunctionSet):
    def __init__(self, **layers):
        super(VRAE, self).__init__(**layers)

    def softplus(self, x):
        return F.log(F.exp(x) + 1)

    def binary_cross_entropy(self, y, t):
        ya = y.data
        ta = t.data
        d  = -1*np.sum(ya*np.log(ta) + (1-ya)*np.log(1-ta))
        v_d = Variable(np.array(d).astype(np.float32))
        return v_d

    def forward_one_step(self, x_data, state, continuous=True, nonlinear_q='tanh', nonlinear_p='tanh', output_f = 'sigmoid', gpu=-1):

        output = np.zeros( x_data.shape ).astype(np.float32)

        nonlinear = {'sigmoid': F.sigmoid, 'tanh': F.tanh, 'softplus': self.softplus, 'relu': F.relu}
        nonlinear_f_q = nonlinear[nonlinear_q]
        nonlinear_f_p = nonlinear[nonlinear_p]

        output_a_f = nonlinear[output_f]

        # compute q(z|x)
        for i in range(x_data.shape[0]):
            x_in_t = Variable(x_data[i].reshape((1, x_data.shape[1])))
            hidden_q_t = nonlinear_f_q( self.recog_in_h( x_in_t ) + self.recog_h_h( state['recog_h'] ) )
            state['recog_h'] = hidden_q_t

        q_mean = self.recog_mean( state['recog_h'] )
        q_log_sigma = 0.5 * self.recog_log_sigma( state['recog_h'] )

        eps = np.random.normal(0, 1, q_log_sigma.data.shape ).astype(np.float32)

        if gpu >= 0:
            eps = cuda.to_gpu(eps)

        eps = Variable(eps)
        z   = q_mean + F.exp(q_log_sigma) * eps

        # compute p( x | z)

        h0 = nonlinear_f_p( self.z(z) )
        out= self.output(h0)
        x_0 = output_a_f( out )
        state['gen_h'] = h0
        if gpu >= 0:
            np_x_0 = cuda.to_cpu(x_0.data)
            output[0] = np_x_0
        else:
            output[0] = x_0.data

        if continuous == True:
            rec_loss = F.mean_squared_error(x_0, Variable(x_data[0].reshape((1, x_data.shape[1]))))
        else:
            rec_loss = F.sigmoid_cross_entropy(out, Variable(x_data[0].reshape((1, x_data.shape[1])).astype(np.int32)))

        x_t = x_0

        for i in range(1, x_data.shape[0]):
            h_t_1 = nonlinear_f_p( self.gen_in_h( x_t ) + self.gen_h_h(state['gen_h']) )
            x_t_1      = self.output(h_t_1)
            state['gen_h'] = h_t_1

            if continuous == True:
                output_t   = output_a_f( x_t_1 )
                rec_loss += F.mean_squared_error(output_t, Variable(x_data[i].reshape((1, x_data.shape[1]))))

            else:
                out = x_t_1
                rec_loss += F.sigmoid_cross_entropy(out, Variable(x_data[i].reshape((1,x_data.shape[1])).astype(np.int32)))
                x_t = output_t = output_a_f( x_t_1 )

            if gpu >= 0:
                np_output_t = cuda.to_cpu(output_t.data)
                output[i] = np_output_t
            else:
                output[i]  = output_t.data


        KLD = -0.0005 * F.sum(1 + q_log_sigma - q_mean**2 - F.exp(q_log_sigma))

        return output, rec_loss, KLD, state


    def generate_z_x(self, seq_length_per_z, sample_z, nonlinear_q='tanh', nonlinear_p='tanh', output_f='sigmoid', gpu=-1):

        output = np.zeros((seq_length_per_z * sample_z.shape[0], self.recog_in_h.W.shape[1]))

        nonlinear = {'sigmoid': F.sigmoid, 'tanh': F.tanh, 'softplus': self.softplus, 'relu': F.relu}
        nonlinear_f_q = nonlinear[nonlinear_q]
        nonlinear_f_p = nonlinear[nonlinear_p]

        output_a_f = nonlinear[output_f]

        for epoch in xrange(sample_z.shape[0]):
            gen_out = np.zeros((seq_length_per_z, output.shape[1]))
            z = Variable(sample_z[epoch].reshape((1, sample_z.shape[1])))

            # compute p( x | z)
            h0 = nonlinear_f_p( self.z(z) )
            x_gen_0 = output_a_f( self.output(h0) )
            state['gen_h'] = h0
            if gpu >= 0:
                np_x_gen_0 = cuda.to_cpu(x_gen_0.data)
                gen_out[0] = np_x_gen_0
            else:
                gen_out[0] = x_gen_0.data

            x_t_1 = x_gen_0

            for i in range(1, seq_length_per_z):
                hidden_p_t = nonlinear_f_p( self.gen_in_h(x_t_1) + self.gen_h_h(state['gen_h']) )
                output_t   = output_a_f( self.output(hidden_p_t) )
                if gpu >= 0:
                    np_output_t = cuda.to_cpu(output_t.data)
                    gen_out[i] = np_output_t
                else:
                    gen_out[i]  = output_t.data
                state['gen_h'] = hidden_p_t
                x_t_1 = output_t

            output[epoch*seq_length_per_z+1:(epoch+1)*seq_length_per_z, :] = gen_out[1:]

        return output

def make_initial_state(n_units, state_pattern, Train=True):
    return {name: Variable(np.zeros((1, n_units), dtype=np.float32), volatile=not Train) for name in state_pattern}
