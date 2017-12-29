def create_huber_loss(delta):
    from keras import backend as K
    import tensorflow as tf

    def huber_loss(y_true, y_pred):
        err = y_true - y_pred
        cond = K.abs(err) < delta
        L2 = 0.5 * K.square(err)
        L1 = delta * (K.abs(err) - 0.5 * delta)
        loss = tf.where(cond, L2, L1)
        return K.mean(loss)

    return huber_loss
