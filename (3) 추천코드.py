(3.a) 추천코드
class GradientEnhancedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=2.0):
        """
        Args:
            alpha (float): 기본 MSE와 gradient loss 간의 가중치 (0~1)
            beta (float): 급격한 변화에 대한 가중치 증폭 계수
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, y_pred, y_true):
        # 기본 MSE 손실
        mse_loss = torch.mean((y_pred - y_true) ** 2)
        
        # 시간에 따른 변화율 계산
        true_gradient = torch.abs(y_true[1:] - y_true[:-1])
        pred_gradient = torch.abs(y_pred[1:] - y_pred[:-1])
        
        # 급격한 변화에 대한 가중치 계산
        gradient_weights = torch.exp(self.beta * true_gradient)
        gradient_weights = gradient_weights / gradient_weights.mean()  # 정규화
        
        # Gradient loss 계산
        gradient_loss = torch.mean(gradient_weights * (pred_gradient - true_gradient) ** 2)
        
        # 최종 손실 = α * MSE + (1-α) * Gradient Loss
        total_loss = self.alpha * mse_loss + (1 - self.alpha) * gradient_loss
        return total_loss


(사용법)
# alpha와 beta값 조정
criterion = GradientEnhancedLoss(alpha=0.7, beta=1.5)

(3.b) 추천코드
class AsymmetricLoss(nn.Module):
    def __init__(self, threshold=0.1, penalty_factor=2.0):
        """
        Args:
            threshold (float): 급격한 변화로 간주할 임계값
            penalty_factor (float): 급격한 변화에 대한 페널티 계수
        """
        super().__init__()
        self.threshold = threshold
        self.penalty_factor = penalty_factor
        
    def forward(self, y_pred, y_true):
        # 시점 간 변화량 계산
        changes = torch.abs(y_true[1:] - y_true[:-1])
        
        # 급격한 변화 지점 식별
        large_changes = changes > self.threshold
        
        # 예측값과 실제값의 차이
        errors = (y_pred - y_true) ** 2
        
        # 가중치 계산: 급격한 변화 지점에서는 더 큰 패널티
        weights = torch.ones_like(errors)
        weights[1:][large_changes] *= self.penalty_factor
        
        # 가중치가 적용된 MSE 계산
        weighted_loss = torch.mean(weights * errors)
        return weighted_loss

(사용법)
# threshold, penalty_factor 조정
criterion = AsymmetricLoss(threshold=0.1, penalty_factor=2.0)
