import tensorflow.keras.backend as K


class ContrastiveMarginLoss:

    def __init__(self, margin=1.0):

        self.margin = margin

    def __call__(self, y_true, y_pred):

        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(self.margin - y_pred, 0))
        ret = K.mean(y_true * square_pred + (1 - y_true) * margin_square)

        return ret
