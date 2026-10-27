---
title: Ternary Modeller İçin Özel Backward Mekanizması
summary: Ternary modellerde gradyanların bozulması ve türev alınamazlığı nedeniyle geliştirilen, STE için bir alternatif.
status: active
order: 20
owners: []
---

# Ternary Modeller İçin Özel Backward Mekanizması

!!! note "Taslak sayfa"
    Bu sayfa projenin kısa açıklamasından oluşturulmuş bir iskelettir; ayrıntıları proje sorumluları ekleyecek.

## Özet

[Ternary](../resources/glossary.md#ternary) modellerde ağırlıklar ayrık değerlerle temsil edilir ve yuvarlama işlemi
türevlenemez. Bu proje, bu soruna yönelik [Düz Geçiş Tahmincisi (STE)](../resources/glossary.md#ste) için bir alternatif geliştiriyor.

## Neden?

Gradyanların yuvarlama adımından geçebilmesi için STE yuvarlamayı yok sayar. Ancak ternary modellerde
[gradyanlar](../resources/glossary.md#gradient) bozulabiliyor; bu projenin çıkış noktası bu sorun.

## Yaklaşım

_Ayrıntılar proje sorumluları tarafından eklenecek._ Arka plan için
[Bitnet 1.58 çevirisi](../resources/q.md) ve okuma listesindeki
[kuantizasyon bölümü](../resources/learning.md#konu-quantization) faydalıdır.

## Durum

Aktif geliştirme aşamasında.

## Yol haritası

- [ ] Problemin ölçülebilir tanımı
- [ ] Alternatif mekanizmanın ilk uygulaması
- [ ] STE ile karşılaştırma

## Nasıl katılırım?

Katılmak için [Katkı Rehberi](../getting-started/contributing.md)'ni okuyun ve Discord'da proje sorumlularına ulaşın.
