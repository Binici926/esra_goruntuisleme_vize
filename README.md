# esra_goruntuisleme_vize
El İzleme ve Mesafe Hesaplama Projesi Kurulum ve Çalıştırma Rehberi:
Bu proje, MediaPipe kütüphanesini kullanarak gerçek zamanlı olarak el izleme yapmayı, parmaklar arasındaki mesafeleri hesaplamayı ve bu verileri ekranda görselleştirmeyi sağlar.

1. Gereksinimler:Projeyi çalıştırmak için bu yazılımlara ve kütüphanelere ihtiyacımız olacak:
opencv-python - Görüntü işleme için kullanılır.
numpy - Matematiksel işlemler ve diziler için kullanılır.
mediapipe - El izleme ve analiz için kullanılır.
Uygulamanın Çalıştırılması:
esra_goruntuisleme_vize.py Bu komut, uygulamayı başlatacak ve kamera üzerinden el izleme işlemini gerçekleştirecektir. Aşağıda, çalıştırıldıktan sonra görmeyi beklediğimiz bazı şeyler:
Gerçek zamanlı kamera görüntüsü: Kameradan alınan görüntüde, el hareketlerinizi takip edeceksiniz.
Parmaklar arasındaki mesafe: İşaret parmağı ve baş parmak arasındaki mesafe hesaplanacak ve ekranda gösterilecektir.
Sağ/Sol El Bilgisi: Hangi elin sağ ya da sol olduğu ekranda belirecek.
Parlaklık ve Netlik Ayarları: Parmaklar arasındaki mesafe arttıkça, yani parmaklar açıldıkça, görüntüdeki parlaklık ve netlik de artacaktır. Aksine, parmaklar arasındaki mesafe azaldıkça, yani parmaklar birbirine yaklaştıkça, parlaklık ve netlik azalacaktır. Bu özellik, görsel deneyimi daha dinamik hale getirecektir.
Çıkmak için: q tuşuna basarak uygulamayı sonlandırabilirsiniz.
