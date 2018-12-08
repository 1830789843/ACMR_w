import tensorflow as tf
from tensorflow.python.framework import ops
# from tensorflow.python.framework import ops 定义了一个图的节点，用以执行在tensor上的计算


# flip翻转：求minimax时，梯度上升和梯度下降。通过参数来控制。
# 翻转梯度
# from tensorflow.python.framework import ops 自定义op, 梯度

class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls

        # print(grad_name)
        # 定义一个梯度，类型为ops的，名称为grad_name

        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            # print(grad)
            # print(tf.negative(grad))
            return [tf.negative(grad) * l]
        
        g = tf.get_default_graph()
        # print(g)
        with g.gradient_override_map({"Identity": grad_name}):
            # print(x)
            # 返回与input x(tensor)有着相同shape和内容的tensor
            y = tf.identity(x)
            # print(y)
        # print(1)
        # print(y) # Tensor("Identity:0", shape=(), dtype=int32)
            
        self.num_calls += 1
        return y


flip_gradient = FlipGradientBuilder()
# print(flip_gradient.num_calls)
# print(flip_gradient.__call__(111))
