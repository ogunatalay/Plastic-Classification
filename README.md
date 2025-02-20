# Plastic-Classification
# plasticClassification
Bu proje için kullanılan veri setine Kaggle üzerinden ulaşabilirsiniz: https://www.kaggle.com/datasets/remnazkarart/plastic-classification-dataset

### Plastik Sınıflandırma ve Karbon Ayak İzi Tahmini

Bu proje, plastik türlerini sınıflandırmak ve karbon ayak izi tahmini yapmak için çeşitli makine öğrenimi ve derin öğrenme modellerini kullanır. **VGG16 tabanlı transfer öğrenimi**, **CNN**, ve **ANN** modelleri, plastik görüntülerini sınıflandırmak için eğitilmiştir. Ayrıca, **SVM**, **Random Forest** ve **Karar Ağaçları** gibi klasik makine öğrenimi modelleri de uygulanmıştır.

---

### Özellikler
- **Veri Ön İşleme**: Görüntüler yeniden boyutlandırılır, normalleştirilir ve veri artırma teknikleri ile çeşitlendirilir.
- **Model Eğitimi**:
  - **Derin Öğrenme Modelleri**: VGG16, CNN ve ANN, plastik türlerini sınıflandırmak için eğitilir.
  - **Klasik Makine Öğrenimi Modelleri**: VGG16'dan çıkarılan özelliklerle SVM, Random Forest ve Karar Ağaçları kullanılır.
- **Karbon Ayak İzi Tahmini**: Plastik türü tahminine dayalı olarak rastgele karbon ayak izi hesaplanır.

---

### Karbon Ayak İzi Değerleri
Tahmin edilen plastik türüne göre karbon ayak izi değerleri (kg CO2e):
| **Plastik Türü** | **Karbon Ayak İzi (kg CO2e)** |
|-------------------|-------------------------------|
| PET               | 2.8 - 4.2                    |
| HDPE              | 1.8 - 2.0                    |
| PVC               | 1.9 - 3.0                    |
| LDPE              | 1.8 - 2.0                    |
| PP                | 1.7 - 2.0                    |
| PS                | 3.0 - 3.5                    |

---

### Kullanım
1. **Veri Hazırlığı**: Eğitim, doğrulama ve test veri setlerini uygun dizinlerde düzenleyin.
2. **Model Eğitimi**: Kod, derin öğrenme ve klasik makine öğrenimi modellerini eğitir.
3. **Tahmin ve Karbon Ayak İzi Hesaplama**:
   - Test veri seti veya kullanıcı tarafından sağlanan bir görüntü üzerinde tahmin yapabilirsiniz.

---

### Çıktılar
- **Sınıflandırma Raporu ve Karışıklık Matrisi**: Modellerin doğruluğu, F1 skoru ve diğer metrikler.
- **Eğitim Performansı Grafikleri**: Eğitim ve doğrulama doğruluğu/kayıp grafikleri.
- **Karbon Ayak İzi Tahmini**: Tahmin edilen plastik türüne dayalı olarak hesaplanan karbon ayak izi.

---

### İletişim
Herhangi bir soru veya öneriniz için bir issue açabilir veya bana ulaşabilirsiniz:  
**ogun.atalay33@gmail.com**
