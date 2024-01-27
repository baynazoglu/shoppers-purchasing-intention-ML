## shoppers-purchasing-intention-ML
Online Shoppers Purchasing Intention Dataset Machine Learning Project

<img src="https://github.com/baynazoglu/shoppers-purchasing-intention-ML/blob/main/customers.png" alt="Image" width="600" height="420">

# 1.Introduction
E-commerce is a very large business sector which gives shoppers access to a large variety of goods and services with a few clicks.
Many popular shopping platforms such as Amazon or Alibaba process millions of transactions every year.
The online shopping market is very competitive and it is important for online shopping platforms to be robust and innovative.
A possible way to increase online shopping transactions is understanding and responding to the behavior of online shoppers.
Given enough online shopping data and machine learning techniques, it is possible to determine the shopping intent of website visitors
-----------------------------------------------------------------------------------------------------------------------------------------
# 2. Data Understanding
The dataset consists of 10 numerical and 8 categorical attributes.

Administrative - count of pages visited by the visitor (e.g. user details and account)
Administrative_Duration - total time spent (seconds) in on Administrative type of page
Informational - count of pages visited by the visitor (e.g. about and contact of the website)
Informational_Duration - total time spent (seconds) in on Informational type of page
ProductRelated - count of pages visited by the visitor (e.g. product details)
ProductRelated_Duration - total time spent (seconds) in on ProductRelated type of page
BounceRates - percentage of visitors who enter the site from that page and then leave ("bounce") without triggering any other requests to the analytics server
ExitRates - the percentage of visitors to a page on the website from which they exit the website to a different website
PageValues - the average value for a page that a user visited before landing on the goal page
SpecialDay - indicates the closeness of the site visiting time to a specific special day (e.g. Mother’s Day, Valentine's Day)
Month - the month of the visit to the website
OperatingSystems - the type of operation system used by the visitor
Browser - the type of browser used by the visitor
Region - the geographic region from which the session started
TrafficType - describes how traffic arrived on the website (Direct, Organic, Referral, Social, Email, Display and Paid)
VisitorType - returning or new visitor or other
Weekend - indicating whether the date of the visit is weekend
Revenue - indicates whether the visitor made a purchase or not

 """
 Bounce Rate and Exit Rate

 Bounce rate is the overall percentage of a single engagement session whereas exit rate is the percentage of exits from a page.
 Hence the former is calculated by dividing the aggregation of one-page visits to the overall entrance visits whereas latter is calculated by dividing the aggregation of total exits from a page to the total visits to a page.

 One major difference between these closely tied metrics is that exit rate is related to the overall percentage of visitors that were within the last session whereas bounce rates account for the percentage of visitors that were part of that one and only session.
 Hence in the case of bounce rate, prior activity is not considered. Hence all bounces logically define exits but conversely it is not true .

A high bounce rate could indicate issues with user satisfaction 1 owing to one or many reasons such as unfriendly UI of the website,
extremely slow throughput or other technical matters. A high exit rate could be a sign of lower performing sectors in funnels, showing areas open to optimization as if customers are leaving then at the end of the day no one is buying.
According to BigCommerce 2 , A bounce rate between 30% to 55% is acceptable. Our analysis shows the bounce rates 7 largely scattered lower than 10%. According to UpSide Business 3 , a bounce rate lower than 5% is a cause 8 of concern indicating a possibility of the Google Analytics code was inserted twice.
 Hence more investigation is needed on these data. Given that there is indeed no error we could look for ways to optimize bounce rates and exit rates to ensure saving sales and securing customer loyalty.


 """
Attribute Information:

The dataset consists of 10 numerical and 8 categorical attributes. The 'Revenue' attribute can be used as the class label. Of the 12,330 sessions in the dataset, 84.5% (10,422) were negative class samples that did not end with shopping, and the rest (1908) were positive class samples ending with shopping.

"Administrative", "Administrative Duration", "Informational", "Informational Duration", "Product Related" and "Product Related Duration" represent the number of different types of pages visited by the visitor in that session and total time spent in each of these page categories.

The values of these features are derived from the URL information of the pages visited by the user and updated in real time when a user takes an action, e.g. moving from one page to another.

The "Bounce Rate", "Exit Rate" and "Page Value" features represent the metrics measured by "Google Analytics" for each page in the e-commerce site. The value of "Bounce Rate" feature for a web page refers to the percentage of visitors who enter the site from that page and then leave ("bounce") without triggering any other requests to the analytics server during that session.

The value of "Exit Rate" feature for a specific web page is calculated as for all pageviews to the page, the percentage that were the last in the session. The "Page Value" feature represents the average value for a web page that a user visited before completing an e-commerce transaction.

The "Special Day" feature indicates the closeness of the site visiting time to a specific special day (e.g. Mother’s Day, Valentine's Day) in which the sessions are more likely to be finalized with transaction.

The value of this attribute is determined by considering the dynamics of e-commerce such as the duration between the order date and delivery date. For example, for Valentina’s day, this value takes a nonzero value between February 2 and February 12, zero before and after this date unless it is close to another special day, and its maximum value of 1 on February 8.

The dataset also includes operating system, browser, region, traffic type, visitor type as returning or new visitor, a Boolean value indicating whether the date of the visit is weekend, and month of the year.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

TR
# 1.Veri seti hakkında genel bilgiler:

Ana hedef, bir alışverişçinin davranışını tahmin etmeye en çok katkıda bulunan temel ölçütlerin tanımlanması ve
bunlarla ilgili öncelikli kritik öneriler ve performans iyileştirmeleri önermek etrafında dönüyordu.
Gelir(Revenue), bir satın alma işleminin yapılıp yapılmadığını belirleyen ilgi özelliğidir.

bir e-ticaret web sitesinden toplanan verileri içermektedir. Veri seti, 12.330 tekil müşteriye ait
18 farklı özellikten feature oluşmaktadır. Bu özellikler arasında müşteri profili, web sitesi etkileşimleri,
sayfa gezinme özellikleri, sepet içeriği gibi değişkenler bulunmaktadır. Veri setinde ayrıca her müşterinin
sonraki ziyaretinde alışveriş yapma olasılığını tahmin etmek için kullanılan "Revenue" adlı bir hedef değişken
bulunmaktadır.

# 2.Veri setinin kolonları ve açıklamaları:

Bu veri setindeki kolonların isimleri ve açıklamaları şunlardır:

-Administrative: Ziyaretçinin site üzerindeki "admin" sayfalarını ziyaret etme sayısı.
-Administrative_Duration: Ziyaretçinin site üzerinde harcadığı zamanın toplamı.
-Informational: Ziyaretçinin site üzerindeki "bilgilendirici" sayfaları ziyaret etme sayısı.
-Informational_Duration: Ziyaretçinin "bilgilerindirici" sayfalar üzerinde harcadığı zamanın toplamı.
-ProductRelated: Ziyaretçinin site üzerindeki "ürün" sayfalarını ziyaret etme sayısı.
-ProductRelated_Duration: Ziyaretçinin site üzerinde harcadığı zamanın toplamı.
-BounceRates: Web sitesine o sayfadan giren ve herhangi bir ek görevi tetiklemeden çıkan ziyaretçilerin yüzdesi.
-ExitRates: Ziyaretçinin bir sayfadan ayrılma oranı.
-PageValues: Hedef sayfanın değeri ve/veya bir e-Ticaret işleminin tamamlanması üzerinden ortalaması alınan sayfanın ortalama değeri. detaylı bilgi :https://support.google.com/analytics/answer/2695658?hl=en
-SpecialDay: Özel günlerde (Mesela Anneler Günü) site ziyareti yapan kullanıcıların oranı.
-Month: Ziyaretin gerçekleştiği ay.
-OperatingSystems: Kullanılan işletim sistemi. (1:Windows 2:Linux 3:Macintosh 4:iOS 5:Android 6:Other/Unknown)
-Browser: Kullanılan internet tarayıcısı. (50 adet kategori var o yüzden bu kolonda değişiklik yapılmalı)
-Region: Ziyaretçinin coğrafi bölgesi. (1:United States 2:Canada 3:United Kingdom 4:Australia 5:Germany 6:Brazil 7:France 8:India 9:Italy)
-TrafficType: An integer value representing what type of traffic the user is categorized into. https://www.practicalecommerce.com/Understanding-Traffic-Sources-in-Google-Analytics (39 tane var bunda da değişiklik yapılmalı)
-VisitorType: Ziyaretçinin siteye ilk defa mı geldiği yoksa önceden ziyaret ettiği bilgisi.
-Weekend: Ziyaretin haftasonuna denk gelip gelmediği bilgisi.
-Revenue: Ziyaret sonucunda satın alma yapılıp yapılmadığı bilgisi. (Hedef değişken)
 """

------------------------------------------------------------------------------------
""" Hemen çıkma oranı(Bounce Rate):
Tek bir etkileşim oturumunun genel yüzdesidir, çıkış oranı ise bir sayfadan yapılan çıkışların yüzdesidir. Bu nedenle ilki, bir sayfalık ziyaretlerin toplamının genel giriş ziyaretlerine bölünmesiyle hesaplanırken ikincisi, bir sayfadan toplam çıkışların toplamının bir sayfaya yapılan toplam ziyaretlere bölünmesiyle hesaplanır. Birbirine yakından bağlı bu metrikler arasındaki en büyük farklardan biri, çıkış oranının son oturumda bulunan ziyaretçilerin genel yüzdesiyle ilgili olması, hemen çıkma oranlarının ise o tek ve tek oturumun parçası olan ziyaretçilerin yüzdesini oluşturmasıdır. Bu nedenle, hemen çıkma oranı söz konusu olduğunda, önceki etkinlik dikkate alınmaz. Bu nedenle, tüm sıçramalar mantıksal olarak çıkışları tanımlar, ancak bunun tersi doğru değildir.

Yüksek bir hemen çıkma oranı, web sitesinin kullanıcı dostu olmayan kullanıcı arabirimi, son derece yavaş aktarım hızı veya diğer teknik sorunlar gibi bir veya birçok nedenden dolayı kullanıcı memnuniyeti 1 ile ilgili sorunları gösterebilir. Yüksek bir çıkış oranı, dönüşüm hunilerinde daha düşük performans gösteren sektörlerin bir işareti olabilir ve optimizasyona açık alanları, sanki müşteriler ayrılıyor ve günün sonunda kimse satın almıyormuş gibi gösteriyor. BigCommerce 2'ye göre %30 ile %55 arasında bir hemen çıkma oranı kabul edilebilir. Analizimiz, hemen çıkma oranlarının büyük ölçüde %10'un altına dağıldığını gösteriyor. UpSide Business 3'e göre, hemen çıkma oranının %5'in altında olması, Google Analytics kodunun iki kez eklenmiş olma olasılığını gösteren bir endişe nedenidir8. Bu nedenle, bu veriler üzerinde daha fazla araştırmaya ihtiyaç vardır. Gerçekten bir hata olmadığı göz önüne alındığında, satışlardan tasarruf etmek ve müşteri sadakatini güvence altına almak için hemen çıkma oranlarını ve çıkış oranlarını optimize etmenin yollarını arayabiliriz.

Sayısal niteliklerin çoğu yüksek pozitif çarpıklık sergilerken, bazıları nominal olarak negatif çarpıklık sergiliyor.

 """Öznitelik Bilgileri:
Veri seti 10 sayısal ve 8 kategorik özellikten oluşmaktadır. 'Gelir' özniteliği, sınıf etiketi olarak kullanılabilir. Veri setindeki 12.330 oturumun %84,5'i (10.422) alışverişle bitmeyen negatif sınıf örnekleri, geri kalanı (1908) alışverişle biten pozitif sınıf örnekleriydi.

"İdari", "Yönetim Süresi", "Bilgilendirme", "Bilgilendirme Süresi", "Ürünle İlgili" ve "Ürünle İlgili Süre", ziyaretçinin o oturumda ziyaret ettiği farklı sayfa türlerinin sayısını ve her birinde geçirilen toplam süreyi temsil eder. bu sayfa kategorileri.

Bu özelliklerin değerleri, kullanıcı tarafından ziyaret edilen sayfaların URL bilgilerinden elde edilir ve kullanıcı bir işlem yaptığında gerçek zamanlı olarak güncellenir, örn. bir sayfadan diğerine geçmek.

"Hemen Çıkma Oranı", "Çıkış Oranı" ve "Sayfa Değeri" özellikleri, e-ticaret sitesindeki her sayfa için "Google Analytics" tarafından ölçülen metrikleri temsil eder. Bir web sayfası için "Hemen Çıkma Oranı" özelliğinin değeri, o sayfadan siteye giren ve ardından o oturum sırasında analiz sunucusuna başka herhangi bir istek tetiklemeden ayrılan ("hemen çıkma") ziyaretçi yüzdesini ifade eder.

Belirli bir web sayfası için "Çıkış Oranı" özelliğinin değeri, sayfanın tüm sayfa görüntülemeleri için, oturumdaki son sayfa görüntüleme yüzdesi olarak hesaplanır. "Sayfa Değeri" özelliği, bir kullanıcının bir e-ticaret işlemini tamamlamadan önce ziyaret ettiği bir web sayfasının ortalama değerini temsil eder.

"Özel Gün" özelliği, site ziyaret saatinin, oturumların işlemle sonuçlanma olasılığının daha yüksek olduğu belirli bir özel güne (örn. Anneler Günü, Sevgililer Günü) yakınlığını belirtir.

Bu özelliğin değeri, sipariş tarihi ile teslimat tarihi arasındaki süre gibi e-ticaretin dinamikleri dikkate alınarak belirlenir. Örneğin Valentina günü için bu değer 2 Şubat ile 12 Şubat arasında sıfır dışında bir değer alır, başka bir özel güne yakın olmadıkça bu tarihten önce ve sonra sıfır, 8 Şubat'ta ise en yüksek değeri 1'dir.

Veri kümesi ayrıca işletim sistemi, tarayıcı, bölge, trafik türü, geri dönen veya yeni ziyaretçi olarak ziyaretçi türü, ziyaret tarihinin hafta sonu ve yılın ayı olup olmadığını gösteren bir Boole değeri içerir.
"""


