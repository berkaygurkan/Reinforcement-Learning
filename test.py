import tensorflow as tf
from tensorflow.core.util import event_pb2
import os

# --- LÜTFEN GÜNCELLEYİN ---
# Lütfen buradaki dosya adını, kendi log dosyanızın tam adıyla değiştirin.
log_file_name = "events.out.tfevents.1757672171.Berkay-MacBook-Pro.local.6316.0" 
log_dir = "notebooks/chapter_1/logs/DQN_CartPole_1"
log_file_path = os.path.join(log_dir, log_file_name)
# -------------------------

print(f"Kontrol edilen log dosyası: {log_file_path}")

# Dosyanın var olup olmadığını kontrol edelim
if not os.path.exists(log_file_path):
    print("\nHATA: Belirtilen log dosyası bulunamadı. Lütfen dosya yolunu ve adını kontrol edin.")
else:
    try:
        # Log dosyasını okumayı dene
        serialized_examples = tf.data.TFRecordDataset(log_file_path)
        
        print("\nDosya başarıyla açıldı. İçerik taranıyor...")
        
        event_tags = set()
        event_count = 0

        # Dosyadaki her bir olayı (event) döngüye al
        for serialized_example in serialized_examples:
            event = event_pb2.Event.FromString(serialized_example.numpy())
            
            # Sadece 'summary' içeren olayları işle (metriklerin olduğu yer)
            if event.HasField('summary'):
                event_count += 1
                # Her bir değerin etiketini (tag) topla
                for value in event.summary.value:
                    event_tags.add(value.tag)

        if event_count > 0:
            print(f"\n[BAŞARILI] Dosya geçerli görünüyor. Toplam {event_count} adet metrik kaydı bulundu.")
            print("Bulunan metrik etiketleri (tag'ler):")
            for tag in sorted(list(event_tags)):
                print(f"- {tag}")
            print("\nSorun, muhtemelen TensorBoard'un önbellekleme (caching) problemidir.")
            print("Çözüm Önerisi: TensorBoard'u tamamen kapatıp (Ctrl+C), 'logs' klasörünü silip, Jupyter Notebook'u yeniden başlatarak eğitim kodunu tekrar çalıştırın.")
        else:
            print("\n[HATA] Dosya okunabiliyor ancak içerisinde görselleştirilecek metrik (summary) verisi bulunamadı.")
            print("Bu durum, eğitim sırasında bir sorun oluştuğu ve loglamanın düzgün tamamlanmadığı anlamına gelebilir.")
            print("Çözüm Önerisi: Lütfen '10_cartpole_dqn_sb3.ipynb' dosyasını yeniden çalıştırarak log dosyasının tekrar oluşturulmasını sağlayın.")

    except Exception as e:
        print(f"\n[KRİTİK HATA] Log dosyası okunurken bir hata oluştu: {e}")
        print("Dosya bozuk olabilir. Lütfen 'logs' klasörünü tamamen silip eğitimi yeniden başlatın.")