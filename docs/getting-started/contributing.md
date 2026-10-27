# Katkı Rehberi

Deep Star Lab'a katkı sunmak isteyenler için kısa bir yol haritası.

## Wiki İçın Yardım Türleri

- Yeni proje kaydı ekleme
- Makale özeti çıkarma
- Kod dökümantasyonu yardımı
- Kendi içeriklerinizi mirrorlamak
- Mevcut sayfaları güncelleme / çeviri
- Hata düzeltme ve öneri (issue)

## Adımlar

1. Repoyu fork'layın
2. Değişikliklerinizi yapın,paket kurulumunu halledin ve  `mkdocs serve` ile önizle.
3. Değişikliklerinizi pushlayın
4. Pull request aç ve kısa bir açıklama ekle

## Yazar ve çevirmen künyesi

Her kişi `data/authors.yml` içinde **bir kez** tanımlanır (`ad-soyad` kimliği, "Ad Soyad" adı). Sayfalarda isim yazmak yerine
front matter'da kimlik kullanılır; başlığın altındaki künye otomatik oluşur:

```yaml
---
title: Makale başlığı
authors: [ad-soyad, baska-yazar]      # orijinal yazarlar
translators: [berkan-dogan]           # çevirmenler
date_translated: 2026-06-01
original_url: https://...
---
```

Kimlik `authors.yml`'de yoksa derleme açık bir hata verir. Çevirilerde terimler için [Sözlük](../resources/glossary.md)'ü kullanın.
Katkınızın [Liderlik Tablosu](../leaderboard.md)'na girmesi için `data/contributions.yml` dosyasına kanıtıyla bir giriş ekleyin.
Genel [Kurallar](../rules.md) için ilgili sayfaya bakın.

!!! note
    Türkçe içerik varsayılan dildir. Her sayfanın `.en.md` uzantılı
    İngilizce karşılığını eklemeyi unutma.
