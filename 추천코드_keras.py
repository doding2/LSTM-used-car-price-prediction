import tensorflow as tf
from keras.src.losses import Loss

class GradientEnhancedLoss(Loss):
    def __init__(self, alpha=0.5, beta=2.0, **kwargs):
        """
        Args:
            alpha (float): 기본 MSE와 gradient loss 간의 가중치 (0~1)
            beta (float): 급격한 변화에 대한 가중치 증폭 계수
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        # 기본 MSE 손실
        mse_loss = tf.reduce_mean(tf.square(y_pred - y_true))

        # 시간에 따른 변화율 계산
        true_gradient = tf.abs(y_true[1:] - y_true[:-1])
        pred_gradient = tf.abs(y_pred[1:] - y_pred[:-1])

        # 급격한 변화에 대한 가중치 계산
        gradient_weights = tf.exp(self.beta * true_gradient)
        gradient_weights = gradient_weights / tf.reduce_mean(gradient_weights)  # 정규화

        # Gradient loss 계산
        gradient_loss = tf.reduce_mean(gradient_weights * tf.square(pred_gradient - true_gradient))

        # 최종 손실 = α * MSE + (1-α) * Gradient Loss
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * gradient_loss
        return total_loss

class AsymmetricLoss(Loss):
    def __init__(self, threshold=0.1, penalty_factor=2.0, **kwargs):
        """
        Args:
            threshold (float): 급격한 변화로 간주할 임계값
            penalty_factor (float): 급격한 변화에 대한 페널티 계수
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.penalty_factor = penalty_factor

    def call(self, y_true, y_pred):
        # 시점 간 변화량 계산
        changes = tf.abs(y_true[1:] - y_true[:-1])

        # 급격한 변화 지점 식별
        large_changes = tf.where(changes > self.threshold)

        # 예측값과 실제값의 차이
        errors = tf.square(y_pred - y_true)

        # 가중치 초기화
        weights = tf.ones_like(errors)

        # 가중치 계산: 급격한 변화 지점에서는 더 큰 페널티
        updates = tf.fill(tf.shape(large_changes)[0:1], self.penalty_factor)
        weights = tf.tensor_scatter_nd_update(weights, large_changes, updates)

        # 가중치가 적용된 MSE 계산
        weighted_loss = tf.reduce_mean(weights * errors)
        return weighted_loss