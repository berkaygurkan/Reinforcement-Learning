# src/envs/wrappers.py

import gymnasium as gym

class RewardShapeWrapper(gym.Wrapper):
    """
    Sopanın açısı belirli bir eşiği aştığında ajana ceza (negatif ödül)
    ekleyerek ödül fonksiyonunu şekillendiren bir sarmalayıcı.
    """
    def __init__(self, env: gym.Env, angle_threshold: float = 0.1, penalty: float = 0.5):
        """
        Yapıcı metot.
        
        Args:
            env (gym.Env): Sarılacak olan Gymnasium ortamı.
            angle_threshold (float): Cezanın uygulanacağı radyan cinsinden açı eşiği.
            penalty (float): Uygulanacak ceza miktarı (ödülden çıkarılacak pozitif değer).
        """
        super().__init__(env)
        self.angle_threshold = angle_threshold
        self.penalty = penalty
    
    def step(self, action):
        """
        Ortamın step fonksiyonunu modifiye eder.
        """
        # Orijinal ortamdan bir sonraki adımı al
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Gözlem vektöründen sopanın açısını al (3. eleman, index 2)
        pole_angle = obs[2]
        
        # Kendi ödül mantığımızı uygula
        if abs(pole_angle) > self.angle_threshold:
            reward -= self.penalty
            
        return obs, reward, terminated, truncated, info