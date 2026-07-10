# Transfer Learning ile Zatürre Hastalığı Tespiti

Bu proje, göğüs röntgeni (X-Ray) görüntülerini analiz ederek zatürre (pneumonia) hastalığını otomatik olarak tespit etmek amacıyla geliştirilmiş bir derin öğrenme projesidir. Projede, önceden eğitilmiş güçlü bir bilgisayarlı görü modeli olan **DenseNet121** mimarisi kullanılarak transfer öğrenme (transfer learning) yöntemi uygulanmıştır.

---

## Veri Seti

Projede kullanılan veri setine aşağıdaki bağlantıdan ulaşabilirsiniz:

- **Kaggle:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

Veri seti; **Normal** ve **Zatürre (Pneumonia)** olmak üzere iki ana sınıftan oluşan göğüs röntgeni görüntülerine sahiptir. Eğitim (train), doğrulama (val) ve test (test) klasörleri şeklinde ayrılmıştır.

---

## Kullanılan Teknolojiler & Kütüphaneler

- **Programlama Dili:** Python
- **Derin Öğrenme Frameworkü:** TensorFlow / Keras
- **Veri Analizi & Görselleştirme:** NumPy, Pandas, Matplotlib, Seaborn
- **Model Mimarisi:** DenseNet121 (Transfer Learning)

---

## Proje Yapısı

```text
Pneumonia-Detection-Transfer-Learning/
│
├── data/               # Kaggle'dan indirilen veri seti klasörleri (Git tarafından takip edilmez)
│
├── src/                # Proje kaynak kodları
│   └── transfer_learning.py
│
├── .gitignore          # Büyük dosyaların ve sanal ortamın engellenmesi için ayarlar
├── README.md           # Proje hakkında genel bilgiler
└── requirements.txt    # Gerekli kütüphanelerin listesi
```
