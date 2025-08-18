import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EmbeddingAlignLoss(nn.Module):
    """소스별 임베딩 분포를 제어하는 손실 함수"""
    def __init__(self, embedding_dim=512, momentum=0.9, temperature=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.momentum = momentum
        self.temperature = temperature
        
        # 현재 epoch의 임베딩 정보를 누적하는 버퍼
        self.register_buffer('current_samitrop_sum', torch.zeros(embedding_dim))
        self.register_buffer('current_negative_sum', torch.zeros(embedding_dim))
        self.register_buffer('current_samitrop_count', torch.tensor(0))
        self.register_buffer('current_negative_count', torch.tensor(0))
        
        # 이전 epoch의 평균 임베딩 (실제 손실 계산에 사용)
        self.register_buffer('prev_samitrop_center', torch.zeros(embedding_dim))
        self.register_buffer('prev_negative_center', torch.zeros(embedding_dim))
        
        # 현재 epoch 번호 추적
        self.register_buffer('current_epoch', torch.tensor(0))
        self.register_buffer('centers_available', torch.tensor(False))
        
    def start_new_epoch(self):
        """새로운 epoch 시작 시 호출"""
        # 이전 epoch의 누적된 정보로 평균 계산
        if self.current_samitrop_count > 0:
            self.prev_samitrop_center = self.current_samitrop_sum / self.current_samitrop_count
        if self.current_negative_count > 0:
            self.prev_negative_center = self.current_negative_sum / self.current_negative_count
        
        # 현재 epoch 정보 초기화
        self.current_samitrop_sum.zero_()
        self.current_negative_sum.zero_()
        self.current_samitrop_count.zero_()
        self.current_negative_count.zero_()
        
        # epoch 번호 증가
        self.current_epoch += 1
        
        # 2번째 epoch부터 center 사용 가능
        if self.current_epoch >= 2:
            self.centers_available = torch.tensor(True)
            
        print(f"[EmbeddingAlignLoss] Epoch {self.current_epoch} started. Centers available: {self.centers_available}")
        if self.centers_available:
            print(f"[EmbeddingAlignLoss] SAMITROP center norm: {torch.norm(self.prev_samitrop_center):.4f}")
            print(f"[EmbeddingAlignLoss] Negative center norm: {torch.norm(self.prev_negative_center):.4f}")
        
    def accumulate_embeddings(self, embeddings, sources):
        """배치에서 임베딩 정보를 누적"""
        with torch.no_grad():
            # 소스별 마스크 생성    
            samitrop_mask = torch.tensor(
                [s == 'samitrop' for s in sources], 
                dtype=torch.bool, 
                device=embeddings.device
            )
            negative_mask = torch.tensor(
                [s in ['ptbxl', 'code15'] for s in sources], 
                dtype=torch.bool, 
                device=embeddings.device
            )

            # SAMITROP 임베딩 누적
            if samitrop_mask.any():
                samitrop_embeddings = embeddings[samitrop_mask]
                self.current_samitrop_sum += samitrop_embeddings.sum(dim=0)
                self.current_samitrop_count += samitrop_embeddings.size(0)
            
            # Negative 임베딩 누적
            if negative_mask.any():
                negative_embeddings = embeddings[negative_mask]
                self.current_negative_sum += negative_embeddings.sum(dim=0)
                self.current_negative_count += negative_embeddings.size(0)
    
    def forward(self, embeddings, sources, alpha=0.5, beta=0.1):
        """
        embeddings: (batch_size, embedding_dim) - 모델의 임베딩
        sources: (batch_size,) - 소스 정보 리스트
        alpha: 소스 내 응집도 가중치
        beta: 소스 간 분리도 가중치
        """
        # 현재 배치의 임베딩 정보 누적
        self.accumulate_embeddings(embeddings, sources)
        
        # 첫 번째 epoch에서는 손실 계산하지 않음
        if not self.centers_available:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        align_loss = 0.0
        
        # L2 정규화
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        samitrop_center_norm = F.normalize(self.prev_samitrop_center, p=2, dim=0)
        negative_center_norm = F.normalize(self.prev_negative_center, p=2, dim=0)
        
        # 소스별 마스크
        samitrop_mask = torch.tensor(
            [s == 'samitrop' for s in sources], 
            dtype=torch.bool, 
            device=embeddings.device
        )
        negative_mask = torch.tensor(
            [s in ['ptbxl', 'code15'] for s in sources], 
            dtype=torch.bool, 
            device=embeddings.device
        )

        # 1. 소스 내 응집도 (Intra-source cohesion)
        if samitrop_mask.any():
            samitrop_embeddings = embeddings_norm[samitrop_mask]
            samitrop_distances = 1 - torch.mm(samitrop_embeddings, samitrop_center_norm.unsqueeze(1)).squeeze(1)
            samitrop_cohesion_loss = samitrop_distances.mean()
            align_loss += alpha * samitrop_cohesion_loss
        
        if negative_mask.any():
            negative_embeddings = embeddings_norm[negative_mask]
            negative_distances = 1 - torch.mm(negative_embeddings, negative_center_norm.unsqueeze(1)).squeeze(1)
            negative_cohesion_loss = negative_distances.mean()
            align_loss += alpha * negative_cohesion_loss
        
        # 2. 소스 간 분리도 (Inter-source separation)
        center_similarity = torch.dot(samitrop_center_norm, negative_center_norm)
        separation_loss = torch.max(torch.tensor(0.0, device=embeddings.device), 
                                  center_similarity + 0.5)  # margin = 0.5
        align_loss += beta * separation_loss
        
        return align_loss
    
    def get_epoch_stats(self):
        """현재 epoch의 통계 정보 반환"""
        stats = {
            'current_epoch': self.current_epoch.item(),
            'samitrop_count': self.current_samitrop_count.item(),
            'negative_count': self.current_negative_count.item(),
            'centers_available': self.centers_available.item()
        }
        
        if self.centers_available:
            stats.update({
                'samitrop_center_norm': torch.norm(self.prev_samitrop_center).item(),
                'negative_center_norm': torch.norm(self.prev_negative_center).item(),
                'center_similarity': torch.dot(
                    F.normalize(self.prev_samitrop_center, p=2, dim=0),
                    F.normalize(self.prev_negative_center, p=2, dim=0)
                ).item()
            })
        
        return stats


class SourceAwareBCELoss(nn.Module):
    """소스별 가중치를 적용한 BCE 손실"""
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        
    def forward(self, outputs, labels, sources):
        # 기본 BCE 손실
        bce_loss = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=self.pos_weight)
        
        # 소스별 가중치 적용
        source_weights = torch.ones_like(labels)
        
        for i, source in enumerate(sources):
            if source == 'samitrop':
                source_weights[i] *= 2.5  # samitrop은 신뢰도가 높으므로 가중치 증가
            elif source == 'code15':
                source_weights[i] *= 0.8  # code15는 레이블이 부정확할 수 있으므로 가중치 감소
            elif source == 'ptbxl':
                source_weights[i] *= 7.5  # ptbxl은 신뢰도가 높으므로 가중치 증가

        weighted_bce_loss = F.binary_cross_entropy_with_logits(
            outputs, labels, weight=source_weights, pos_weight=self.pos_weight
        )
        
        return 0.6 * bce_loss + 0.4 * weighted_bce_loss

