# Transfer Learning ile Akciğer Röntgenlerinden Zatürre (Pneumonia) Teşhisi

Bu proje, derin öğrenme (Deep Learning) yöntemiyle göğüs röntgeni (X-Ray) görüntülerinden otomatik olarak zatürre teşhisi koyabilen bir **Bilgisayarlı Görü (Computer Vision)** sistemidir. Projede, önceden milyonlarca genel resimle eğitilmiş güçlü **DenseNet121** mimarisi kullanılarak **Transfer Learning (Ağırlık Aktarımı)** tekniği uygulanmıştır.

---

## Proje Mimarisi ve Akışı

Projenin arkasındaki akış, görüntü hazırlığından son teşhis kararına kadar şu 3 temel aşamadan oluşmaktadır:

1. **Görüntü İşleme (Image Processing):** Ham röntgen resimleri $224 \times 224$ boyutuna getirilmiş, pikseller $[0, 1]$ arasına sıkıştırılarak (normalizasyon) modelin işlem yapabileceği temiz matrislere dönüştürülmüştür.
2. **Derin Öğrenme (Deep Learning) Motoru:** Sınıflandırma yeteneği yüksek olan **DenseNet121** modelinin ilk katmanları dondurulmuş (`base_model.trainable = False`), böylece mevcut medikal örüntü yakalama gücü korunmuştur. Modelin ucuna tıbbi karar vermesi için şu özel katmanlar eklenmiştir:
   - **GlobalAveragePooling2D:** Çok boyutlu matrisleri düzleştirerek tek boyutlu özellik vektörüne dönüştürür.
   - **Dense (128, ReLU):** Detaylı tıbbi örüntü analizlerini yapan ara gizli katmandır.
   - **Dropout (0.5):** Modelin eğitim verisini ezberlemesini (`overfitting`) engellemek için eğitim sırasında nöronların %50'sini rastgele kapatan güvenlik kilididir.
   - **Dense (1, Sigmoid):** En sondaki tek karar verici nörondur. Matematiksel sinyalleri $0$ ile $1$ arasında bir zatürre olasılığına dönüştürür.
3. **Bilgisayarlı Görü (Computer Vision) Kararı:** Eğitilen bu model, test setindeki röntgenleri analiz ederek en nihayetinde klinik bir karar ("Sağlıklı" veya "Zatürre") üretir.

---

## Teknik Özellikler ve Parametreler

- **Base Model:** DenseNet121 (ImageNet ağırlıklarıyla başlatıldı, üst katmanları donduruldu)
- **Optimizer:** Adam (Öğrenme Hızı: `1e-4` / `0.0001` - Hassas transfer learning için küçük tutulmuştur)
- **Kayıp Fonksiyonu (Loss):** Binary Crossentropy (İkili sınıflandırma kaybı)
- **Performans Metrikleri:** Accuracy (Doğruluk) ve Recall (Duyarlılık)
  - _Not:_ Tıp projelerinde gerçekten hasta olanları kaçırmamak kritik olduğundan, gözümüzün hiçbir zatürre vakasını kaçırmamasını sağlayan **Recall** metriğine özel önem verilmiştir.

### Otomatik Eğitim Kontrolü (Callbacks)

Modelin eğitim esnasında en iyi performansı göstermesi için şu geri çağırma (callbacks) yapıları kurulmuştur:

- **EarlyStopping (Sabır = 3):** Doğrulama kaybı (`val_loss`) 3 epoch boyunca iyileşmezse eğitimi ezberi önlemek adına otomatik durdurur ve en iyi ağırlıkları geri yükler.
- **ReduceLROnPlateau (Sabır = 2, Faktör = 0.2):** Model öğrenirken tıkandığı an öğrenme hızını dinamik olarak düşürerek daha hassas adımlarla optimize olmasını sağlar.
- **ModelCheckpoint:** Eğitim süresince elde edilen en başarılı modeli otomatik olarak `best_model.h5` adıyla kaydeder.

---

## Sonuç Değerlendirme (Confusion Matrix)

Eğitim tamamlandıktan sonra, modelin test seti üzerindeki tahmin doğruluğunu ve hata tiplerini analiz etmek amacıyla **Karmaşıklık Matrisi (Confusion Matrix)** çizdirilmektedir.

Bu matris sayesinde modelin:

- Kaç hastayı doğru teşhis ettiği (**True Positive**)
- Kaç sağlıklı insanı doğru ayırt ettiği (**True Negative**)
- En önemlisi, kaç hasta insanı gözden kaçırdığı (**False Negative**) net bir şekilde gözlemlenebilmektedir.

---

## Kurulum ve Çalıştırma

Projenin sanal ortamını (venv) aktif hale getirdikten sonra modeli eğitmek ve test analizlerini görmek için terminalde şu komutu çalıştırmanız yeterlidir:

```bash
python transfer_learning.py
```
