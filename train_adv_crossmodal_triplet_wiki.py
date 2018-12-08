import tensorflow as tf
from models.adv_crossmodal_triplet_wiki import AdvCrossModalSimple, ModelParams
# from models.wiki_shallow import AdvCrossModalSimple, ModelParams


def main(_):
    graph = tf.Graph()
    model_params = ModelParams()
    model_params.update()

    # 建立模型
    with graph.as_default():
        model = AdvCrossModalSimple(model_params)
    # 训练并评估
    with tf.Session(graph=graph) as sess:
        model.train(sess)
        # model.eval_random_rank()
        model.eval(sess)


if __name__ == '__main__':
    tf.app.run()
