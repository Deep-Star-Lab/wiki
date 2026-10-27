# Liderlik Tablosu

Deep Star Lab araştırmacılarının katkıları. Sıralama **puana** göre yapılır; puanlar
aşağıdaki kurallardan, katkı defterindeki kayıtlara bakılarak otomatik hesaplanır.

<!-- dsl:leaderboard top=50 -->

## Nasıl puan toplanır?

<!-- dsl:scoring -->

- Bir katkıda birden fazla kişi varsa puan **eşit bölünür**.
- “Aylık üst sınır” olan türlerde, bir kişi o türden ayda sınırın üstünde puan alamaz (spam'i engeller).
- İnceleyen kişi kaliteye göre `weight` çarpanı (0.5–2.0) verebilir.
- **Bu ay** ve **Son 90 gün** sekmeleri yeni katılanların da görünür olmasını sağlar; tüm zamanlar tablosu emeği korur.

## Son katkılar

<!-- dsl:ledger n=15 -->

## Katkını nasıl eklersin?

1. Kendin `data/authors.yml` içinde yoksan oraya `ad-soyad` kimliğinle eklen.
2. `data/contributions.yml` dosyasına yeni bir giriş ekle (tür, başlık, tarih ve **kanıt bağlantısı** zorunlu).
3. Pull request aç. İnceleyen, kanıtı doğrulayıp birleştirdiğinde katkın tabloya girer.

```yaml
- people: [ad-soyad]
  type: translation        # paper | release | project | translation | code | summary | docs | review | issue
  title: "Kısa açıklama"
  date: 2026-10-10
  url: https://github.com/Deep-Star-Lab/wiki/pull/123
```

Kayıtlar sitenin her derlemesinde okunur; hatalı bir kimlik, tür ya da tarih derlemeyi
açık bir hata mesajıyla durdurur.
